# Copyright (c) OpenMMLab. All rights reserved.
from .cross_entropy_loss import (
    CrossEntropyLoss,
    binary_cross_entropy,
    cross_entropy,
    mask_cross_entropy,
)
from .change_loss import ChangeLoss
from .cross_entropy_softmax import SoftmaxCrossEntropyLoss
from .dice_loss import DiceLoss
from .lovasz_loss import LovaszLoss
from .utils import reduce_loss, weight_reduce_loss, weighted_loss
from ._builder import LOSSES,build_losses

__all__ = [
    "cross_entropy",
    "binary_cross_entropy",
    "mask_cross_entropy",
    "CrossEntropyLoss",
    "reduce_loss",
    "weight_reduce_loss",
    "weighted_loss",
    "LovaszLoss",
    "DiceLoss",
    "SoftmaxCrossEntropyLoss",
]
