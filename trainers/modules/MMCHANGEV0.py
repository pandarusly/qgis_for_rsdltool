from abc import ABC
from typing import Any, Dict

import torch
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader

# from mmseg.core import add_prefix  ‘.’
from mmseg.models import build_segmentor
from trainers.utils.lr_scheduler_torch import build_scheduler
from trainers.utils.optimizer import build_optimizer
from .msic import build_metric, add_prefix  # '/'
from benchmarks.MixOC import resize


class MMCHANGEV0(LightningModule, ABC):
    def __init__(
            self,
            Config,
            HPARAMS_LOG=False,
            example_input_array=(1, 3, 256, 256),
            CKPT=False,
            **kwargs
    ):
        super().__init__()

        Config = OmegaConf.create(Config)
        self.save_hyperparameters(
            Config, logger=HPARAMS_LOG)  # True在hydra下运行不起来
        self.lr = self.hparams.TRAIN.BASE_LR
        self.example_input_array = (
            torch.randn(*example_input_array),
            torch.randn(*example_input_array),
        )

        self._load(self.hparams.MODEL)
        self.train_metrics, self.val_metrics, self.test_metrics = build_metric(use_torch=False,
                                                                               num_classes=self.num_classes)

        if CKPT:
            self._finetue(CKPT)

    def _load(self, model):
        if isinstance(model, DictConfig):
            model = OmegaConf.to_object(model)
        self.model = build_segmentor(model)
        # self.model.init_weights()
        self.num_classes = self.model.num_classes

    def _finetue(self, ckpt_path):
        print("-" * 30)
        print("locate new momdel pretrained {}".format(ckpt_path))
        print("-" * 30)
        pretained_dict = torch.load(ckpt_path)["state_dict"]
        self.load_state_dict(pretained_dict)

    def forward(self, img1, img2):
        x = self.model.extract_feat(img1, img2)
        seg_logits = self.model.decode_head(x)
        if self.model.with_auxiliary_head:
            result = [seg_logits]
            if isinstance(self.model.auxiliary_head, nn.ModuleList):
                for idx, aux_head in enumerate(self.model.auxiliary_head):
                    result.append(aux_head(x))
            else:
                result.append(self.model.auxiliary_head(x))
            return result
        return seg_logits

    def forward_dummy(self, img1, img2):
        """
        为了计算flops
        """
        return self.model.forward_dummy(img1, img2)

    def cal_losses(self, seg_logits, seg_label):
        losses = dict()
        if isinstance(seg_logits, (list, tuple)):
            main_seg = seg_logits[0]
            loss_decode = self.model.decode_head.losses(main_seg, seg_label)
            losses.update(add_prefix(loss_decode, "decode"))
            if isinstance(self.model.auxiliary_head, nn.ModuleList):
                for idx, aux_head in enumerate(self.model.auxiliary_head):
                    loss_aux = aux_head.losses(seg_logits[idx+1], seg_label)
                    losses.update(add_prefix(loss_aux, f"aux_{idx}"))
            else:
                loss_aux = self.model.auxiliary_head.losses(
                    seg_logits[1], seg_label)
                losses.update(add_prefix(loss_aux, f"aux_{idx}"))
        else:
            loss_decode = self.model.decode_head.losses(seg_logits, seg_label)
            losses.update(add_prefix(loss_decode, "decode"))
        return losses

    def configure_optimizers(self):
        optimizer = build_optimizer(self.hparams, self.parameters())
        scheduler = build_scheduler(self.hparams, optimizer)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=2)
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "interval": "epoch",
        #     },
        # }
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.hparams.TRAIN.INTERVAL,
                "monitor": self.hparams.TRAIN.MONITOR
            },
        }

    # def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
    #     # print("\ncurrent_epoch ", self.current_epoch)
    #     # print("global_step", self.global_step)
    #     scheduler.step(
    #         epoch=self.current_epoch
    #     )  # timm's scheduler need the epoch value

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        mask = batch["gt_semantic_seg"]  # b 1 h w
        x, y = batch["img1"], batch["img2"]  # b c h w
        seg_logits = self(x, y)
        losses = self.cal_losses(seg_logits, mask)
        loss, log_vars = self.model._parse_losses(losses)
        # loss, log_vars = _parse_losses(loss_change)
        log_vars = add_prefix(log_vars, "train")
        # -------- log losses
        self.log_dict(log_vars, on_step=False, on_epoch=True, prog_bar=False)
        # self.log_dict(log_vars, on_step=True,
        #               on_epoch=False, prog_bar=False)
        # ------- log f1,iou
        if isinstance(seg_logits, (list, tuple)):
            seg_logits = seg_logits[0]
        seg_logits = resize(
            input=seg_logits,
            size=mask.shape[2:],
            mode="bilinear",
            align_corners=self.model.align_corners,
        )
        preds = seg_logits.argmax(1)

        if mask.ndim == 4:
            mask = mask.squeeze(1)

        self.train_metrics(preds, mask)

        return {"loss": loss}

    def training_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level training metrics.

        Args:
            outputs: list of items returned by training_step
        """
        metrics = self.train_metrics.compute()
        log_vars = add_prefix(metrics, "train")
        self.log_dict(log_vars, prog_bar=False, sync_dist=True)
        self.train_metrics.reset()

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:

        mask = batch["gt_semantic_seg"]  # b 1 h w
        x, y = batch["img1"], batch["img2"]  # b c h w
        seg_logits = self(x, y)
        losses = self.cal_losses(seg_logits, mask)
        loss, log_vars = self.model._parse_losses(losses)
        log_vars = add_prefix(log_vars, "val")
        # -------- log losses

        self.log_dict(log_vars, on_step=False, on_epoch=True, prog_bar=False)

        # ------- log f1,iou
        if isinstance(seg_logits, (list, tuple)):
            seg_logits = seg_logits[0]
        seg_logits = resize(
            input=seg_logits,
            size=mask.shape[2:],
            mode="bilinear",
            align_corners=self.model.align_corners,
        )
        preds = seg_logits.argmax(1)

        if mask.ndim == 4:
            mask = mask.squeeze(1)

        self.val_metrics(preds, mask)

        return {"loss": loss}

    def validation_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level training metrics.

        Args:
            outputs: list of items returned by training_step
        """
        metrics = self.val_metrics.compute()
        log_vars = add_prefix(metrics, "val")
        self.log_dict(log_vars, prog_bar=True, sync_dist=True)
        self.val_metrics.reset()

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:

        mask = batch["gt_semantic_seg"]  # b 1 h w
        x, y = batch["img1"], batch["img2"]  # b c h w
        seg_logits = self(x, y)
        # ------- log f1,iou
        if isinstance(seg_logits, (list, tuple)):
            seg_logits = seg_logits[0]
        seg_logits = resize(
            input=seg_logits,
            size=mask.shape[2:],
            mode="bilinear",
            align_corners=self.model.align_corners,
        )
        preds = seg_logits.argmax(1)
        if mask.ndim == 4:
            mask = mask.squeeze(1)

        self.test_metrics(preds, mask)

    def test_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level training metrics.

        Args:
            outputs: list of items returned by training_step
        """
        metrics = self.test_metrics.compute()
        log_vars = add_prefix(metrics, "test")
        self.log_dict(log_vars, prog_bar=True, sync_dist=True)
        self.test_metrics.reset()
