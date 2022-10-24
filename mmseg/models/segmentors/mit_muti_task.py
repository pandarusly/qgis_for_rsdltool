# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from mmseg.core import add_prefix
from mmseg.ops import resize
from .base import BaseSegmentor
from .. import builder
from ..builder import SEGMENTORS


@SEGMENTORS.register_module()
class MitMutiTask(BaseSegmentor):
    """Encoder Decoder segmentors for ChangeDetection.
    """

    def __init__(self,
                 backbone=dict(
                     type='mit_b0',
                     strides=[4, 2, 2, 2]
                 ),
                 decode_head=dict(
                     type='UPerHead',
                     in_channels=[32, 64, 160, 256],
                     in_index=[0, 1, 2, 3],
                     pool_scales=(2, 3, 6),
                     channels=128,
                     dropout_ratio=0.0,
                     num_classes=2,
                     align_corners=False,
                     loss_decode=dict(
                         type='ChangeLoss', loss_weight=1.0, loss_name='loss_ce'),
                     norm_cfg=dict(type='BN', requires_grad=True)
                     # norm_cfg=dict(type='SyncBN', requires_grad=True)
                 ),
                 change=dict(
                     type="BaseChange",
                     in_channels=[32, 64, 160, 256],
                     out_channels=[32, 64, 160, 256],
                     fusion_forms=("concate", "concate", "concate", "concate"),
                     last_conv=True,
                     kernel_size=1
                 ),
                 neck=None,
                 auxiliary_head=dict(type="FCCDNHead", in_channels=32),
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):

        super(MitMutiTask, self).__init__(init_cfg)

        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone['pretrained'] = pretrained

        self.backbone = builder.build_backbone(backbone)

        self.change = builder.build_change(change)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def extract_feat(self, img1, img2):
        """Extract features from images."""
        """Extract features from images."""
        x1 = self.backbone(img1)
        x2 = self.backbone(img2)
        if self.with_neck:
            x1 = self.neck(x1)
            x2 = self.neck(x2)

        return x1, x2

    def encode_decode(self, img1, img2, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x1, x2 = self.extract_feat(img1, img2)
        x = self.change(x1, x2)
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img1.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode, seg_logits = self.decode_head.forward_train(x, img_metas,
                                                                 gt_semantic_seg,
                                                                 self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses, seg_logits

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self, x, _attns1, _attns2, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux, _ = aux_head.forward_train(x, _attns1, _attns2, img_metas, gt_semantic_seg,
                                                     self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux, _ = self.auxiliary_head.forward_train(
                x, _attns1, _attns2, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def _auxiliary_head_show(self, img1, img2, ):

        x1, x2 = self.extract_feat(img1, img2)

        if isinstance(self.auxiliary_head, nn.ModuleList):
            a_pred, b_pred = self.auxiliary_head[0].forward(x1, x2)
        else:
            a_pred, b_pred = self.auxiliary_head.forward(x1, x2)

        a_pred = resize(
            input=a_pred,
            size=img1.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        b_pred = resize(
            input=b_pred,
            size=img1.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        a_pred, b_pred = a_pred.argmax(1), b_pred.argmax(1)

        return a_pred, b_pred

    def forward_dummy(self, img1, img2):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img1, img2, None)

        return seg_logit

    def forward_train(self, img1, img2, img_metas, gt_semantic_seg):

        x1, x2 = self.extract_feat(img1, img2)
        x = self.change(x1, x2)

        losses = dict()

        loss_decode, seg_logits = self._decode_head_forward_train(x, img_metas,
                                                                  gt_semantic_seg)

        seg_logits = resize(
            input=seg_logits,
            size=img1.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, x1, x2, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        return losses, seg_logits

    # 不实例化就出错

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        pass

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        pass

    def get_param_groups(self):

        param_groups = [[], [], [], []]  # backbone; backbone_norm; change; (decode_head,neck,auxiliary_head)

        for name, param in list(self.backbone.named_parameters()):

            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        for param in list(self.change.parameters()):
            param_groups[2].append(param)

        for param in list(self.decode_head.parameters()):
            param_groups[3].append(param)

        if self.with_auxiliary_head:

            for param in list(self.auxiliary_head.parameters()):
                param_groups[3].append(param)

        if self.with_neck:
            for param in list(self.neck.parameters()):
                param_groups[3].append(param)

        return param_groups
