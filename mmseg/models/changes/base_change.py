import torch.nn as nn
from mmcv.runner import BaseModule

from . import FusionHEAD
# from .vit_change import VVIT

from ..builder import CHANGES, build_plugin

from benchmarks._builder import CHANGES as CHANGESV2


@CHANGES.register_module(name="BaseChange")
class BaseChange(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 enhance=None,
                 enhance_index=-1,
                 fusion_forms=("concate", "concate", "concate", "concate"),
                 **kwargs
                 ):
        super(BaseChange, self).__init__()
        self.enhance = build_plugin(enhance)
        self.enhance_index = enhance_index

        self.fusion = nn.ModuleList()

        for in_c, out_c, forms in zip(in_channels, out_channels, fusion_forms):
            fusion_heas = FusionHEAD(in_channels=in_c, channels=out_c, fusion_form=forms, **kwargs)

            self.fusion.append(fusion_heas)

    @property
    def with_enhance(self):
        """bool: whether the segmentor has neck"""
        return hasattr(self, 'enhance') and self.enhance is not None

    def forward(self, x1, x2):
        x1, x2 = list(x1), list(x2)

        if self.with_enhance:
            x1[self.enhance_index], x2[self.enhance_index] = self.enhance(x1[self.enhance_index],
                                                                          x2[self.enhance_index])
        out = []
        for i, fuse in enumerate(self.fusion):
            out.append(fuse(x1[i], x2[i]))
        return out


# @CHANGES.register_module()
class STMChange(BaseChange):
    def __init__(self, vit_cfg, **kwargs):
        # enhance = VVIT(**vit_cfg)

        super(STMChange, self).__init__(enhance=enhance, **kwargs)
        assert self.with_enhance

    def forward(self, x1, x2):
        x1, x2 = list(x1), list(x2)
        x1 = x1[self.enhance_index]
        x2 = x2[self.enhance_index]
        out = []
        token, enhance_x1, enhance_x2 = self.enhance(x1, x2)
        out.append(token)
        out.append(self.fusion[self.enhance_index](enhance_x1, enhance_x2))
        return out


@CHANGES.register_module(name="SimpleChange")
class SimpleChange(BaseModule):

    def __init__(self,
                 change_heads,
                 **kwargs
                 ):
        super(SimpleChange, self).__init__()

        self.fusion = nn.ModuleList()

        for change_head in change_heads:
            changeHead = CHANGES.build(change_head)
            self.fusion.append(changeHead)

    def forward(self, x1, x2):
        x1, x2 = list(x1), list(x2)
        out = []
        for i, fuse in enumerate(self.fusion):
            # print(x1[i].shape)
            # print(x2[i].shape)
            out.append(fuse(x1[i], x2[i]))
        return out
