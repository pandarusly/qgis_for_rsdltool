import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.builder import LOSSES


def cross_entropy(
    input,
    target,
    weight=None,
    reduction="mean",
    avg_factor=None,
    class_weight=None,
    ignore_index=255,
):
    """
    logSoftmax_with_loss
    :param reduction:
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C Sample-wise loss weight.
    :return: torch.Tensor [0]
    """
    # print(input.shape)
    # print(target.shape)
    input = input.float()
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(
            input, size=target.shape[1:], mode="bilinear", align_corners=True
        )

    return F.cross_entropy(
        input=input,
        target=target,
        weight=class_weight,
        ignore_index=ignore_index,
        reduction=reduction,
    )


@LOSSES.register_module()
class SoftmaxCrossEntropyLoss(nn.Module):
    def __init__(
        self, class_weight=None, loss_weight=1.0, loss_name="loss_softmaxce", **kwargs
    ):
        super(SoftmaxCrossEntropyLoss, self).__init__()
        self.cls_criterion = cross_entropy
        self.loss_weight = loss_weight
        self.class_weight = self.get_class_weight(class_weight)

        self._loss_name = loss_name

    def forward(self, cls_score, label, weight=None, **kwargs):

        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score, label, weight, class_weight=class_weight, **kwargs
        )
        return loss_cls

    @staticmethod
    def get_class_weight(class_weight):
        """Get class weight for loss function.

        Args:
            class_weight (list[float] | str | None): If class_weight is a str,
                take it as a file name and read from it.
        """
        if isinstance(class_weight, str):
            # take it as a file path
            if class_weight.endswith(".npy"):
                class_weight = np.load(class_weight)
            else:
                # pkl, json or yaml
                class_weight = mmcv.load(class_weight)

        return class_weight

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
