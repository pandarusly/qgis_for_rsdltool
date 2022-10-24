
from abc import ABC

from typing import Any, Dict

import torch
import torch.nn as nn
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning.core.lightning import LightningModule


from benchmarks import build_benchmarks, build_losses

# global_step ,global_step
from trainers.utils.lr_scheduler_torch import build_scheduler
from trainers.utils.optimizer import build_optimizer
from .msic import build_metric, cal_losses, _parse_losses, add_prefix, align_size


class BaseChangeLite(LightningModule, ABC):
    def __init__(self,
                 Config,
                 HPARAMS_LOG=False,
                 example_input_array=(1, 3, 256, 256),
                 CKPT=False,
                 loss_decode=dict(type="ChangeLoss",
                                  loss_name='loss_ce', loss_weight=1.0),
                 **kwargs):
        super().__init__()
        # self.load_from_checkpoint()
        Config = OmegaConf.create(Config)
        self.save_hyperparameters(
            Config, logger=HPARAMS_LOG)  # True在hydra下运行不起来
        self.lr = self.hparams.TRAIN.BASE_LR

        if example_input_array:
            self.example_input_array = (torch.randn(
                *example_input_array), torch.randn(*example_input_array))

        self._load(self.hparams.MODEL)
        self.train_metrics, self.val_metrics, self.test_metrics = build_metric()

        # --------
        if isinstance(loss_decode, dict):
            self.loss_decode = build_losses(loss_decode)
        elif isinstance(loss_decode, DictConfig):
            loss_decode = OmegaConf.to_object(loss_decode)
            self.loss_decode = build_losses(loss_decode)
        elif isinstance(loss_decode, (list, tuple)):
            self.loss_decode = nn.ModuleList()
            for loss in loss_decode:
                self.loss_decode.append(build_losses(loss))
        else:
            raise TypeError(
                f"loss_decode must be a dict or sequence of dict,\
                but got {type(loss_decode)}"
            )
        # --------
        self.model.init_weights()

        if CKPT:
            self._finetue(CKPT)

    def _finetue(self, ckpt_path):
        print("-" * 30)
        print("locate new momdel pretrained {}".format(ckpt_path))
        print("-" * 30)
        pretained_dict = torch.load(ckpt_path)["state_dict"]
        self.load_state_dict(pretained_dict)

    def criterion(self, pred, mask):
        loss = 0.
        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode

        for loss_decode in losses_decode:
            loss += loss_decode(
                pred,
                mask,
                weight=None,
                ignore_index=255)

        return loss

    def change_loss(self, pred, mask, weight=1.0):
        if pred.ndim == mask.ndim:
            mask = mask.squeeze(1)
        loss = dict()
        loss['loss_change'] = weight*self.criterion(pred, mask)
        return loss

    def aux_loss(self, pred, mask, weight=0.4):
        if pred.ndim == mask.ndim:
            mask = mask.squeeze(1)

        if pred.shape[-2:] != mask.shape[-2:]:
            align_size(pred, mask)

        loss = dict()
        loss['loss_aux'] = weight*self.criterion(pred, mask)
        return loss

    def _load(self, model):
        if isinstance(model, DictConfig):
            model = OmegaConf.to_object(model)
        self.model = build_benchmarks(model)
        self.num_classes = 2

    def forward(self, img1, img2):
        # return self.model(img1, img2)
        return self.forward_dummy(img1, img2)

    def forward_dummy(self, img1, img2):
        """
        为了计算flops
        """

        pred = self.model(img1, img2)

        if isinstance(pred, (list, tuple)):
            return align_size(pred[0], img1)
        else:
            return align_size(pred, img1)

    def configure_optimizers(self):
        optimizer = build_optimizer(
            self.hparams, parameters=self.model.parameters())
        scheduler = build_scheduler(self.hparams, opt=optimizer)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=2)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.hparams.TRAIN.INTERVAL,
                "monitor":  self.hparams.TRAIN.MONITOR
            },
        }

    def training_step(
            self, batch: Dict[str, Any], batch_idx: int
    ):
        mask = batch["gt_semantic_seg"]  # b 1 h w
        x, y = batch["img1"], batch["img2"]  # b c h w
        seg_logits = self(x, y)
        # ------- losss
        if isinstance(seg_logits, (list, tuple)):
            loss_change = dict()
            main_seg = seg_logits[0]
            loss_decode = cal_losses(
                main_seg, mask, self.loss_decode, loss_weight=1, align_corners=False, ignore_index=255)
            loss_change.update(add_prefix(loss_decode, "mian"))
            for idx in range(1, len(seg_logits)):
                loss_aux = cal_losses(
                    seg_logits[idx], mask, self.loss_decode, loss_weight=0.2, align_corners=False, ignore_index=255)
                loss_change.update(add_prefix(loss_aux, f"aux_{idx}"))
        else:
            loss_change = cal_losses(
                seg_logits, mask, self.loss_decode, align_corners=False, ignore_index=255)

        loss, log_vars = _parse_losses(loss_change)
        log_vars = add_prefix(log_vars, "train")
        # ------- losss
        self.log_dict(log_vars, on_step=False,
                      on_epoch=True, prog_bar=False)
        # ------- log f1,iou
        if isinstance(seg_logits, (list, tuple)):
            seg_logits = seg_logits[0]

        seg_logits = align_size(seg_logits, mask)
        if mask.ndim == 4:
            mask = mask.squeeze(1)
        self.train_metrics(seg_logits.argmax(1), mask)
        metrics = self.train_metrics.compute()
        log_vars = add_prefix(metrics, "train")
        self.log_dict(log_vars, on_step=False, prog_bar=False, on_epoch=True)
        self.train_metrics.reset()
        return {"loss": loss}

    def training_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level training metrics.

        Args:
            outputs: list of items returned by training_step
        """
        pass
        # metrics = self.train_metrics.compute()
        # log_vars = add_prefix(metrics, "train")
        # self.log_dict(log_vars, prog_bar=False, sync_dist=True)
        # self.train_metrics.reset()

    def validation_step(
            self, batch: Dict[str, Any], batch_idx: int
    ) -> None:

        mask = batch["gt_semantic_seg"]  # b 1 h w
        x, y = batch["img1"], batch["img2"]  # b c h w
        seg_logits = self(x, y)
        # ------- losss
        # ------- losss
        if isinstance(seg_logits, (list, tuple)):
            loss_change = dict()
            main_seg = seg_logits[0]
            loss_decode = cal_losses(
                main_seg, mask, self.loss_decode, loss_weight=1, align_corners=False, ignore_index=255)
            loss_change.update(add_prefix(loss_decode, "mian"))
            for idx in range(1, len(seg_logits)):
                loss_aux = cal_losses(
                    seg_logits[idx], mask, self.loss_decode, loss_weight=0.2, align_corners=False, ignore_index=255)
                loss_change.update(add_prefix(loss_aux, f"aux_{idx}"))
        else:
            loss_change = cal_losses(
                seg_logits, mask, self.loss_decode, align_corners=False, ignore_index=255)
        _, log_vars = _parse_losses(loss_change)
        log_vars = add_prefix(log_vars, "val")
        # ------- losss
        self.log_dict(log_vars, on_step=False,
                      on_epoch=True, prog_bar=False)
        # ------- log f1,iou
        if isinstance(seg_logits, (list, tuple)):
            seg_logits = seg_logits[0]

        seg_logits = align_size(seg_logits, mask)
        if mask.ndim == 4:
            mask = mask.squeeze(1)
        self.val_metrics(seg_logits.argmax(1), mask)

        # # ------- log f1,iou
        # print(log_vars)
        # metrics = self.val_metrics.compute()
        # log_vars = add_prefix(metrics, "val")
        # print(log_vars)

    def validation_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level training metrics.

        Args:
            outputs: list of items returned by training_step
        """
        metrics = self.val_metrics.compute()
        log_vars = add_prefix(metrics, "val")
        self.log_dict(log_vars, prog_bar=True, sync_dist=True)
        self.val_metrics.reset()

    def test_step(
            self, batch: Dict[str, Any], batch_idx: int
    ) -> None:

        mask = batch["gt_semantic_seg"]  # b 1 h w
        x, y = batch["img1"], batch["img2"]  # b c h w
        seg_logits = self.forward_dummy(x, y)
        # ------- log f1,iou
        if mask.ndim == 4:
            mask = mask.squeeze(1)
        # seg_logits = align_size(seg_logits,mask)
        self.test_metrics(seg_logits.argmax(1), mask)

    def test_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level training metrics.

        Args:
            outputs: list of items returned by training_step
        """
        metrics = self.test_metrics.compute()
        log_vars = add_prefix(metrics, "test")
        self.log_dict(log_vars, prog_bar=True, sync_dist=True)
        self.test_metrics.reset()
