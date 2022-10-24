from typing import Any, Dict

import torch
import torch.nn as nn

from benchmarks.losses.dice_loss import DiceLoss
from .BaseChange import BaseChangeLite
# global_step ,global_step
from .msic import _parse_losses, add_prefix, align_size


# class BinaryDiceLoss(nn.Module):
#     """Dice loss of binary class
#     """

#     def __init__(self, smooth=1, p=2, reduction='mean'):
#         super(BinaryDiceLoss, self).__init__()
#         self.smooth = smooth
#         self.p = p
#         self.reduction = reduction

#     def forward(self, predict, target):
#         assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
#         predict = predict.contiguous().view(predict.shape[0], -1)
#         target = target.contiguous().view(target.shape[0], -1)

#         num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
#         den = torch.sum(predict.pow(self.p) +
#                         target.pow(self.p), dim=1) + self.smooth

#         loss = 1 - num / den

#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         elif self.reduction == 'none':
#             return loss
#         else:
#             raise Exception('Unexpected reduction {}'.format(self.reduction))


# class DiceLoss(nn.Module):
#     """Dice loss, need one hot encode input
#     """

#     def __init__(self, weight=None, ignore_index=None, **kwargs):
#         super(DiceLoss, self).__init__()
#         self.kwargs = kwargs
#         self.weight = weight
#         self.ignore_index = ignore_index

#     def forward(self, predict, target):
#         assert predict.shape == target.shape, 'predict & target shape do not match'
#         dice = BinaryDiceLoss(**self.kwargs)
#         total_loss = 0
#         #predict = F.softmax(predict, dim=1)
#         predict = torch.sigmoid(predict)

#         for i in range(target.shape[1]):
#             if i != self.ignore_index:
#                 dice_loss = dice(predict[:, i], target[:, i])
#                 if self.weight is not None:
#                     assert self.weight.shape[0] == target.shape[1], \
#                         'Expect weight shape [{}], get[{}]'.format(
#                             target.shape[1], self.weight.shape[0])
#                     dice_loss *= self.weights[i]
#                 total_loss += dice_loss

#         return total_loss/target.shape[1]


class BCL(nn.Module):
    """
    batch-balanced contrastive loss
    no-change，1
    change，-1
    """

    def __init__(self, margin=2.0):
        super(BCL, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        label[label == 1] = -1
        label[label == 0] = 1

        mask = (label != 255).float()
        distance = distance * mask

        pos_num = torch.sum((label == 1).float())+0.0001
        neg_num = torch.sum((label == -1).float())+0.0001

        loss_1 = torch.sum((1+label) / 2 * torch.pow(distance, 2)) / pos_num
        loss_2 = torch.sum((1-label) / 2 *
                           torch.pow(torch.clamp(
                               self.margin - distance, min=0.0), 2)
                           ) / neg_num
        loss = loss_1 + loss_2
        return loss


class DSAMNLite(BaseChangeLite):
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)

        self.contrative_loss = BCL()
        self.dice_loss = DiceLoss()

    def _finetue(self, ckpt_path):
        print("-" * 30)
        print("locate new momdel pretrained {}".format(ckpt_path))
        print("-" * 30)
        pretained_dict = torch.load(ckpt_path)
        # pretained_dict = torch.load(ckpt_path)["state_dict"]
        self.model.load_state_dict(pretained_dict)

    def get_loss(self, seg_logits, mask):
        loss = dict()
        prob, ds2, ds3 = seg_logits
        # Diceloss
        dsloss2 = self.dice_loss(ds2, mask)
        dsloss3 = self.dice_loss(ds3, mask)

        Dice_loss = 0.5*(dsloss2+dsloss3)

        # contrative loss
        CT_loss = self.contrative_loss(prob, mask)

        # CD loss
        CD_loss = CT_loss + 0.1 * Dice_loss
        loss["CD_loss"] = CD_loss
        return loss

    def make_one_hot(self, input, num_classes=2):
        """Convert class index tensor to one hot encoding tensor.

        Args:
            input: A tensor of shape [N, 1, *]
            num_classes: An int of number of class
        Returns:
            A tensor of shape [N, num_classes, *]
        """
        one_hot = torch.FloatTensor(input.size()[0], num_classes, input.size()[
                                    2], input.size()[3]).zero_().to(input.device)

        target = one_hot.scatter_(1, input.data, 1)
        print(input.data)
        return target

    def training_step(
        self, batch: Dict[str, Any], batch_idx: int
    ):
        mask = batch["gt_semantic_seg"]  # b 1 h w

        x, y = batch["img1"], batch["img2"]  # b c h w
        seg_logits = self(x, y)

        # ------- losss
        loss_change = self.get_loss(seg_logits, mask)
        loss, log_vars = _parse_losses(loss_change)
        log_vars = add_prefix(log_vars, "train")
        # ------- losss
        self.log_dict(log_vars, on_step=False,
                      on_epoch=True, prog_bar=False)
        # ------- log f1,iou

        seg_logits = align_size(seg_logits[0], mask)
        seg_logits = (seg_logits > 1).int()
        if mask.ndim == 4:
            mask = mask.squeeze(1)
        self.train_metrics(seg_logits, mask)
        # # ------- log f1,iou
        # print(log_vars)
        # metrics = self.train_metrics.compute()
        # log_vars = add_prefix(metrics, "train")
        # print(log_vars)

        return {"loss": loss}

    def validation_step(
            self, batch: Dict[str, Any], batch_idx: int
    ) -> None:

        mask = batch["gt_semantic_seg"]  # b 1 h w

        x, y = batch["img1"], batch["img2"]  # b c h w
        seg_logits = self(x, y)

        # ------- losss
        loss_change = self.get_loss(seg_logits, mask)
        loss, log_vars = _parse_losses(loss_change)
        log_vars = add_prefix(log_vars, "val")

        # ------- losss
        self.log_dict(log_vars, on_step=False,
                      on_epoch=True, prog_bar=False)
        # ------- log f1,iou
        seg_logits = align_size(seg_logits[0], mask)
        seg_logits = (seg_logits > 1).int()
        if mask.ndim == 4:
            mask = mask.squeeze(1)
        self.val_metrics(seg_logits, mask)

    def test_step(
            self, batch: Dict[str, Any], batch_idx: int
    ) -> None:

        mask = batch["gt_semantic_seg"]  # b 1 h w
        x, y = batch["img1"], batch["img2"]  # b c h w
        seg_logits = self(x, y)

        seg_logits = align_size(seg_logits[0], mask)
        seg_logits = (seg_logits > 1).int()
        if mask.ndim == 4:
            mask = mask.squeeze(1)
        self.test_metrics(seg_logits, mask)
