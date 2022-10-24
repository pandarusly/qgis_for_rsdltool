from abc import ABC
from collections import OrderedDict
from typing import Any, Dict

import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader, Dataset
from torchmetrics import (
    MetricCollection,
    Accuracy,
    FBetaScore,
    Precision,
    Recall,
)

from benchmarks.ChangeFormer import resize
from benchmarks.FCCDN import FCCDN
from benchmarks.losses.change_loss import ChangeLoss
from trainers.utils.binary import ConfuseMatrixMeter
# global_step ,global_step
from trainers.utils.lr_scheduler_torch import build_scheduler
from trainers.utils.optimizer import build_optimizer


# ----

# Copyright (c) OpenMMLab. All rights reserved.


def add_prefix(inputs, prefix):
    """Add prefix for dict.

    Args:
        inputs (dict): The input dict with str keys.
        prefix (str): The prefix to add.

    Returns:

        dict: The dict with keys updated with ``prefix``.
    """

    outputs = dict()
    for name, value in inputs.items():
        outputs[f'{prefix}.{name}'] = value

    return outputs


def _parse_losses(losses):
    """Parse the raw outputs (losses) of the network.

    Args:
        losses (dict): Raw output of the network, which usually contain
            losses and other necessary information.

    Returns:
        tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
            which may be a weighted sum of all losses, log_vars contains
            all the variables to be sent to the logger.
    """
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                f'{loss_name} is not a tensor or list of tensors')

    loss = sum(_value for _key, _value in log_vars.items()
               if 'loss' in _key)

    log_vars['loss'] = loss
    for loss_name, loss_value in log_vars.items():
        # reduce loss when distributed training
        if dist.is_available() and dist.is_initialized():
            loss_value = loss_value.data.clone()
            dist.all_reduce(loss_value.div_(dist.get_world_size()))
        log_vars[loss_name] = loss_value.item()

    return loss, log_vars

# ------


def build_metric(use_torch=False, num_classes=2):
    if use_torch:
        train_metrics = MetricCollection(
            {
                "acc": Accuracy(),
                "precision_1": Precision(
                    num_classes=num_classes,
                    ignore_index=0,
                    average="macro",
                    mdmc_average="global",
                ),
                "recall_1": Recall(
                    num_classes=num_classes,
                    ignore_index=0,
                    average="macro",
                    mdmc_average="global",
                ),
                "F1_1": FBetaScore(
                    num_classes=num_classes,
                    ignore_index=0,
                    average="macro",
                    mdmc_average="global",
                ),
            }
        )

        val_metrics = train_metrics.clone()
        test_metrics = train_metrics.clone()
    else:
        train_metrics = ConfuseMatrixMeter(num_classes)
        val_metrics = ConfuseMatrixMeter(num_classes)
        test_metrics = ConfuseMatrixMeter(num_classes)
    return train_metrics, val_metrics, test_metrics


class RandomDictDataset(Dataset):
    def __init__(self, size: tuple, num_classes: int, length: int):
        self.len = length
        self.size = size
        self.num_classes = num_classes

    def __getitem__(self, index):
        img1 = torch.randn(self.size)
        img2 = torch.randn(self.size)
        gt_semantic_seg = torch.randint(
            0, self.num_classes, size=(1, self.size[-2], self.size[-1]))
        return {"image": (img1, img2), "mask": gt_semantic_seg}

    def __len__(self):
        return self.len


class FCCDNLite(LightningModule, ABC):
    def __init__(self, Config, HPARAMS_LOG=False, example_input_array=(1, 3, 256, 256), CKPT=False, **kwargs):
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

        # self.loss_decode = dice_bce_loss()
        self.loss_decode = nn.ModuleList()
        # self.loss_decode.append(CrossEntropyLoss(
        #     avg_non_ignore=True, use_sigmoid=False))
        self.loss_decode.append(ChangeLoss(loss_weight=1, loss_name='loss_ce'))

        self.loss_decode_aux = nn.ModuleList()
        self.loss_decode_aux.append(ChangeLoss(loss_weight=0.2))

        # init_weights(self.model)
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

    def aux_criterion(self, a_pred, b_pred_mask_nochange_label, ignore_index=255):
        loss = 0.
        if not isinstance(self.loss_decode_aux, nn.ModuleList):
            losses_decode = [self.loss_decode_aux]
        else:
            losses_decode = self.loss_decode_aux

        for loss_decode in losses_decode:
            loss += loss_decode(
                a_pred,
                b_pred_mask_nochange_label,
                weight=None,
                ignore_index=ignore_index)
        return loss

    def fccdn_loss(self, a_pred, b_pred, gt_semantic_seg):

        loss = dict()
        """ for binary change detection task"""
        if a_pred.shape[-1] != gt_semantic_seg.shape[-1]:
            if gt_semantic_seg.ndim != 4:
                gt_semantic_seg = gt_semantic_seg.unsqueeze(1)
            gt_semantic_seg = resize(
                input=gt_semantic_seg.float(),
                size=a_pred.shape[-2:])

        gt_semantic_seg = gt_semantic_seg.squeeze(1).to(torch.long)

        a_pred_mask_nochange_label = a_pred.clone().detach().argmax(1)
        a_pred_mask_change_label = a_pred.clone().detach().argmax(1)
        b_pred_mask_nochange_label = b_pred.clone().detach().argmax(1)
        b_pred_mask_change_label = b_pred.clone().detach().argmax(1)

        a_pred_mask_nochange_label[gt_semantic_seg == 1] = 255
        b_pred_mask_nochange_label[gt_semantic_seg == 1] = 255

        a_pred_mask_change_label = gt_semantic_seg - a_pred_mask_change_label
        a_pred_mask_change_label[gt_semantic_seg == 0] = 255
        b_pred_mask_change_label = gt_semantic_seg - b_pred_mask_change_label
        b_pred_mask_change_label[gt_semantic_seg == 0] = 255

        loss['loss_bg1'] = self.aux_criterion(a_pred, b_pred_mask_nochange_label,
                                              ignore_index=255)  # 前时相预测不变化的部分要与后时相不变保持一致
        # 后时相预测不变化的部分要与前时相不变保持一致
        loss['loss_bg2'] = self.aux_criterion(
            b_pred, a_pred_mask_nochange_label, ignore_index=255)

        loss['loss_fg1'] = self.aux_criterion(a_pred, b_pred_mask_change_label,
                                              ignore_index=255)  # 前时相预测不变化的部分要与后时相不变保持一致
        # 后时相预测不变化的部分要与前时相不变保持一致
        loss['loss_fg2'] = self.aux_criterion(
            b_pred, a_pred_mask_change_label, ignore_index=255)
        return loss

    def change_loss(self, pred, mask):
        # if pred.ndim == mask.ndim:
        mask = mask.squeeze(1)
        loss = dict()
        loss['loss_change'] = self.criterion(pred, mask)
        return loss

    def _load(self, model):
        if isinstance(model, DictConfig):
            model = OmegaConf.to_object(model)
        self.model = FCCDN(**model)
        self.num_classes = 2

    def forward(self, img1, img2):
        seg_logits = self.model(img1, img2)
        return seg_logits

    def forward_dummy(self, img1, img2):
        """
        为了计算flops
        """
        # pred = self.model(img1, img2)
        pred = self.model(img1, img2)[0]
        return pred
        # pred = (pred.sigmoid()>=0.5).int()
        # return  pred

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

    # import pdb
    # pdb.set_trace()
    def training_step(
            self, batch: Dict[str, Any], batch_idx: int
    ):
        mask = batch["gt_semantic_seg"]  # b 1 h w
        x, y = batch["img1"], batch["img2"]  # b c h w
        losses = dict()

        # seg_logits = self(x, y)
        seg_logits = self.forward(x, y)
        # ------- losss
        y = seg_logits[0]
        loss_change = self.change_loss(y, mask)
        losses.update(loss_change)

        if len(seg_logits) == 3:
            y1, y2 = seg_logits[1], seg_logits[2]

            aux_loss = self.fccdn_loss(y1, y2, mask)
            losses.update(aux_loss)

        loss, log_vars = _parse_losses(losses)
        log_vars = add_prefix(log_vars, "train/")

        self.log_dict(log_vars, on_step=False,
                      on_epoch=True, prog_bar=False)

        # ------- log f1,iou
        if mask.ndim == 4:
            mask = mask.squeeze(1)
        self.train_metrics(y.argmax(1), mask)

        return {"loss": loss}

    def training_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level training metrics.

        Args:
            outputs: list of items returned by training_step
        """
        metrics = self.train_metrics.compute()
        log_vars = add_prefix(metrics, "train/")
        self.log_dict(log_vars, prog_bar=False, sync_dist=True)
        self.train_metrics.reset()

    def validation_step(
            self, batch: Dict[str, Any], batch_idx: int
    ) -> None:

        mask = batch["gt_semantic_seg"]  # b 1 h w
        x, y = batch["img1"], batch["img2"]  # b c h w
        losses = dict()

        seg_logits = self(x, y)
        # ------- losss
        y = seg_logits[0]

        loss_change = self.change_loss(y, mask)
        losses.update(loss_change)
        loss, log_vars = _parse_losses(losses)
        log_vars = add_prefix(log_vars, "val/")

        self.log_dict(log_vars, on_step=False,
                      on_epoch=True, prog_bar=False)

        # ------- log f1,iou
        if mask.ndim == 4:
            mask = mask.squeeze(1)
        self.val_metrics(y.argmax(1), mask)

        # return {"loss": loss}

    def validation_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level training metrics.

        Args:
            outputs: list of items returned by training_step
        """
        metrics = self.val_metrics.compute()
        log_vars = add_prefix(metrics, "val/")
        self.log_dict(log_vars, prog_bar=True, sync_dist=True)
        self.val_metrics.reset()

    def test_step(
            self, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        mask = batch["gt_semantic_seg"]  # b 1 h w
        x, y = batch["img1"], batch["img2"]  # b c h w
        seg_logits = self(x, y)
        # -------- log losses
        y = seg_logits[0]
        # ------- log f1,iou
        if mask.ndim == 4:
            mask = mask.squeeze(1)
        self.test_metrics(y.argmax(1), mask)

    def test_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level training metrics.

        Args:
            outputs: list of items returned by training_step
        """
        metrics = self.test_metrics.compute()
        log_vars = add_prefix(metrics, "test/")
        self.log_dict(log_vars, prog_bar=True, sync_dist=True)
        self.test_metrics.reset()


class BoringFCCDNLite(FCCDNLite):
    def __init__(self, size=(3, 256, 256), **kwargs):
        self.size = size
        super(BoringFCCDNLite, self).__init__(**kwargs)

    def train_dataloader(self):
        return DataLoader(RandomDictDataset(size=self.size, num_classes=self.num_classes, length=64), batch_size=2)

    def val_dataloader(self):
        return DataLoader(RandomDictDataset(size=self.size, num_classes=self.num_classes, length=32), batch_size=2)

    def test_dataloader(self):
        return DataLoader(RandomDictDataset(size=self.size, num_classes=self.num_classes, length=64), batch_size=2)

    def predict_dataloader(self):
        return DataLoader(RandomDictDataset(size=self.size, num_classes=self.num_classes, length=64), batch_size=2)
