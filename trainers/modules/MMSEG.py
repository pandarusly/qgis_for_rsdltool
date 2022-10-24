from abc import ABC

import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader, Dataset

from mmseg.core import add_prefix
from mmseg.models import build_segmentor
from typing import Any, Dict
from trainers.utils.binary import ConfuseMatrixMeter
from trainers.utils.lr_scheduler import build_scheduler
from trainers.utils.optimizer import build_optimizer
from torchmetrics import (
    MetricCollection,
    Accuracy,
    FBetaScore,
    Precision,
    Recall,
)


def build_metric(use_torch=True, num_classes=2):
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


# ============


class RandomDictDataset(Dataset):
    def __init__(self, size: tuple, num_classes: int, length: int):
        self.len = length
        self.size = size
        self.num_classes = num_classes

    def __getitem__(self, index):
        img1 = torch.randn(self.size)
        img2 = torch.randn(self.size)
        gt_semantic_seg = torch.randint(
            0, self.num_classes, size=(1, self.size[-2], self.size[-1])
        )
        img = torch.cat([img1, img2], dim=0)
        return {"img": img, "gt_semantic_seg": gt_semantic_seg}

    def __len__(self):
        return self.len


class SegLitChangeDetection(LightningModule, ABC):
    def __init__(
            self, Config, CKPT=False, HPARAMS_LOG=False, example_input_array=(2, 6, 256, 256), **kwargs
    ):
        super().__init__()

        Config = OmegaConf.create(Config)
        self.save_hyperparameters(
            Config, logger=HPARAMS_LOG)  # True在hydra下运行不起来
        self.lr = self.hparams.TRAIN.BASE_LR
        self.CKPT = CKPT
        self.example_input_array = (
            torch.randn(*example_input_array),
            torch.randn(*example_input_array),
        )

        self._load(self.hparams.MODEL)
        (
            train_metrics,
            val_metrics,
            test_metrics,
        ) = build_metric(use_torch=True, num_classes=self.num_classes)

        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics

        if CKPT:
            self._finetue(CKPT)

    def _load(self, model):
        if isinstance(model, DictConfig):
            model = OmegaConf.to_object(model)
        self.model = build_segmentor(model)
        # if not self.CKPT:
            # self.model.init_weights()
        self.num_classes = self.model.num_classes

    def _finetue(self, ckpt_path):
        print("-" * 30)
        print("locate new momdel pretrained {}".format(ckpt_path))
        print("-" * 30)
        pretained_dict = torch.load(ckpt_path)["state_dict"]
        self.load_state_dict(pretained_dict)

    def forward(self, img, gt_semantic_seg=None, return_loss=False):
        if return_loss:
            return self.model.forward_train(
                img, img_metas=None, gt_semantic_seg=gt_semantic_seg
            )
        else:
            return self.model.forward_dummy(img)

    def forward_dummy(self, img):
        """
        为了计算flops
        """
        return self.model.forward_dummy(img)

    def configure_optimizers(self):
        optimizer = build_optimizer(self.hparams, self.parameters())
        scheduler = build_scheduler(self.hparams, optimizer=optimizer)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=2)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        # print("\ncurrent_epoch ", self.current_epoch)
        # print("global_step", self.global_step)
        scheduler.step(
            epoch=self.current_epoch
        )  # timm's scheduler need the epoch value

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        losses, seg_logits = self.forward(**batch, return_loss=True)
        loss, log_vars = self.model._parse_losses(losses)
        log_vars = add_prefix(log_vars, "train/")
        # -------- log losses
        self.log_dict(log_vars, on_step=False, on_epoch=True, prog_bar=False)
        # self.log_dict(log_vars, on_step=True,
        #               on_epoch=False, prog_bar=False)

        # ------- log f1,iou
        preds = seg_logits.argmax(1)

        mask = batch["gt_semantic_seg"]

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
        log_vars = add_prefix(metrics, "train/")
        self.log_dict(log_vars, prog_bar=False, sync_dist=True)
        self.train_metrics.reset()

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:

        losses, seg_logits = self.forward(**batch, return_loss=True)
        loss, log_vars = self.model._parse_losses(losses)
        log_vars = add_prefix(log_vars, "val/")
        # -------- log losses

        self.log_dict(log_vars, on_step=False, on_epoch=True, prog_bar=False)

        # ------- log f1,iou
        preds = seg_logits.argmax(1)

        mask = batch["gt_semantic_seg"]

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
        log_vars = add_prefix(metrics, "val/")
        self.log_dict(log_vars, prog_bar=True, sync_dist=True)
        self.val_metrics.reset()

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:

        seg_logits = self.forward(**batch, return_loss=False)
        # ------- log f1,iou
        preds = seg_logits.argmax(1)

        mask = batch["gt_semantic_seg"]

        if mask.ndim == 4:
            mask = mask.squeeze(1)

        self.test_metrics(preds, mask)

    def test_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level training metrics.

        Args:
            outputs: list of items returned by training_step
        """
        metrics = self.test_metrics.compute()
        log_vars = add_prefix(metrics, "test/")
        self.log_dict(log_vars, prog_bar=True, sync_dist=True)
        self.test_metrics.reset()


class BoringSegLitChangeDetection(SegLitChangeDetection):
    def __init__(self, size=(3, 256, 256), **kwargs):
        self.size = size
        super(BoringSegLitChangeDetection, self).__init__(**kwargs)

    def train_dataloader(self):
        return DataLoader(
            RandomDictDataset(
                size=self.size, num_classes=self.num_classes, length=64),
            batch_size=2,
        )

    def val_dataloader(self):
        return DataLoader(
            RandomDictDataset(
                size=self.size, num_classes=self.num_classes, length=32),
            batch_size=2,
        )

    def test_dataloader(self):
        return DataLoader(
            RandomDictDataset(
                size=self.size, num_classes=self.num_classes, length=64),
            batch_size=2,
        )

    def predict_dataloader(self):
        return DataLoader(
            RandomDictDataset(
                size=self.size, num_classes=self.num_classes, length=64),
            batch_size=2,
        )
