from benchmarks.MixOC import resize
import torch.nn as nn


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
        outputs[f'{prefix}/{name}'] = value

    return outputs


import torch.distributed as dist
from collections import OrderedDict
import torch


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


from torchmetrics import (MetricCollection, Accuracy,
                          FBetaScore, Precision, Recall)

from trainers.utils.binary import ConfuseMatrixMeter


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


def align_size(pred, mask, align_corners=False):
    if pred.shape[-2:] != mask.shape[-2:]:
        pred = resize(
            input=pred,
            size=mask.shape[2:],
            mode="bilinear",
            align_corners=align_corners,
        )
    return pred


def cal_losses(seg_logit, seg_label, loss_decode, align_corners=False, ignore_index=255):
    """Compute segmentation loss."""
    loss = dict()
    seg_logit = resize(
        input=seg_logit,
        size=seg_label.shape[2:],
        mode="bilinear",
        align_corners=align_corners,
    )
    seg_weight = None
    seg_label = seg_label.squeeze(1)

    if not isinstance(loss_decode, nn.ModuleList):
        losses_decode = [loss_decode]
    else:
        losses_decode = loss_decode
    for loss_decode in losses_decode:
        if loss_decode.loss_name not in loss:
            loss[loss_decode.loss_name] = loss_decode(
                seg_logit,
                seg_label,
                weight=seg_weight,
                ignore_index=ignore_index,
            )
        else:
            loss[loss_decode.loss_name] += loss_decode(
                seg_logit,
                seg_label,
                weight=seg_weight,
                ignore_index=ignore_index,
            )
    return loss
