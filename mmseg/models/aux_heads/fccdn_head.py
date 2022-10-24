import numpy as np
import torch
import torch.nn as nn
from mmcv.runner import BaseModule

from mmseg.ops.wrappers import resize
from ..builder import HEADS, build_loss
from ..losses.change_loss import hybrid_loss, cross_entropy


@HEADS.register_module()
class FCCDNHead(BaseModule):
    def __init__(self, in_channels=16, in_index=0,
                 loss_decode=dict(type='ChangeLoss', loss_weight=0.2, loss_name='loss_ce')):
        super(FCCDNHead, self).__init__()
        self.in_index = in_index

        self.attn_proj = nn.Conv2d(
            in_channels=in_channels, out_channels=2, kernel_size=1, bias=True)

        if isinstance(loss_decode, dict):
            self.loss_decode = build_loss(loss_decode)
        elif isinstance(loss_decode, (list, tuple)):
            self.loss_decode = nn.ModuleList()
            for loss in loss_decode:
                self.loss_decode.append(build_loss(loss))
        else:
            raise TypeError(f'loss_decode must be a dict or sequence of dict,\
                but got {type(loss_decode)}')

        # self.criterion = cross_entropy

        nn.init.kaiming_normal_(self.attn_proj.weight,
                                a=np.sqrt(5), mode="fan_out")

    def criterion(self, a_pred, b_pred_mask_nochange_label, ignore_index=255):
        loss = 0.
        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode

        for loss_decode in losses_decode:
            loss += loss_decode(
                a_pred,
                b_pred_mask_nochange_label,
                weight=None,
                ignore_index=ignore_index)

        return loss

    def forward_train(self, x, _attns1, _attns2, img_metas, gt_semantic_seg, train_cfg):
        loss = dict()
        place_hold = None

        a_pred, b_pred = self.forward(_attns1, _attns2)

        if a_pred.shape[-1] != gt_semantic_seg.shape[-1]:
            gt_semantic_seg = resize(
                input=gt_semantic_seg.float(),
                size=a_pred.shape[2:])

        gt_semantic_seg = gt_semantic_seg.squeeze(1)

        # a_pred_mask = a_pred.clone()
        # b_pred_mask = b_pred.clone()

        # a_pred_mask[:, 1, ...][gt_semantic_seg == 1] = 0  # 前时相只保留不变的部分形状
        # a_pred_mask[:, 0, ...][gt_semantic_seg == 1] = 1  # 前时相只保留不变的部分形状
        #
        # b_pred_mask[:, 1, ...][gt_semantic_seg == 1] = 0  # 后时相只保留不变的部分形状
        # b_pred_mask[:, 0, ...][gt_semantic_seg == 1] = 1  # 后时相只保留不变的部分形状

        # a_pred_mask_label = a_pred_mask.clone().detach().argmax(1)
        # b_pred_mask_label = b_pred_mask.clone().detach().argmax(1)

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

        loss['loss_bg1'] = self.criterion(a_pred, b_pred_mask_nochange_label,
                                          ignore_index=255)  # 前时相预测不变化的部分要与后时相不变保持一致
        # 后时相预测不变化的部分要与前时相不变保持一致
        loss['loss_bg2'] = self.criterion(b_pred, a_pred_mask_nochange_label, ignore_index=255)

        loss['loss_fg1'] = self.criterion(a_pred, b_pred_mask_change_label,
                                          ignore_index=255)  # 前时相预测不变化的部分要与后时相不变保持一致
        # 后时相预测不变化的部分要与前时相不变保持一致
        loss['loss_fg2'] = self.criterion(b_pred, a_pred_mask_change_label, ignore_index=255)

        return loss, place_hold

    def forward(self, _attns1, _attns2):
        """Forward function."""
        _attns1, _attns2 = _attns1[self.in_index], _attns2[self.in_index]

        attn_pred1 = self.attn_proj(_attns1)
        attn_pred2 = self.attn_proj(_attns2)

        return attn_pred1, attn_pred2
