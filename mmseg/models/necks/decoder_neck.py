# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import build_norm_layer

from ..builder import NECKS
from .. import builder
from torch import nn


@NECKS.register_module()
class DecoderNeck(nn.Module):
    """ DecoderNeck
    """

    def __init__(self, decoder):
        super(DecoderNeck, self).__init__()
        self.decode_head = builder.build_head(decoder)
        self.decode_head.conv_seg = nn.Identity()

    def forward(self, inputs):
        outputs = [self.decode_head(inputs)]
        return tuple(outputs)
