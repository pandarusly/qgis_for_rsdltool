from abc import ABCMeta
from typing import Union, List, Dict, Tuple

from mmcv.runner import BaseModule
from mmcv.utils import to_2tuple
from mmcv.cnn import ConvModule as MMConvModule
from torch import nn
import torch

# --------------- FCCDN densecat


bn_mom = 0.0003
"""Implemention of dense fusion module"""


class densecat_cat_add(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(densecat_cat_add, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, out_chn, kernel_size=1, padding=0),
            # SynchronizedBatchNorm2d(out_chn, momentum=bn_mom),
            nn.BatchNorm2d(out_chn, momentum=bn_mom),
            torch.nn.ReLU(inplace=True),

        )

    def forward(self, x, y):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2 + x1)

        y1 = self.conv1(y)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2 + y1)

        return self.conv_out(x1 + x2 + x3 + y1 + y2 + y3)


class densecat_cat_diff(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(densecat_cat_diff, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, out_chn, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_chn, momentum=bn_mom),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2 + x1)

        y1 = self.conv1(y)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2 + y1)
        out = self.conv_out(torch.abs(x1 + x2 + x3 - y1 - y2 - y3))
        return out


class DF_Module(nn.Module):
    def __init__(self, dim_in, dim_out, reduction=True):
        super(DF_Module, self).__init__()
        if reduction:
            self.reduction = torch.nn.Sequential(
                torch.nn.Conv2d(dim_in, dim_in // 2, kernel_size=1, padding=0),
                nn.BatchNorm2d(dim_in // 2, momentum=bn_mom),
                torch.nn.ReLU(inplace=True),
            )
            dim_in = dim_in // 2
        else:
            self.reduction = None
        self.cat1 = densecat_cat_add(dim_in, dim_out)
        self.cat2 = densecat_cat_diff(dim_in, dim_out)
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        if self.reduction is not None:
            x1 = self.reduction(x1)
            x2 = self.reduction(x2)
        x_add = self.cat1(x1, x2)
        x_diff = self.cat2(x1, x2)
        y = self.conv1(x_diff) + x_add
        return y


# -------------------------ASPP3dFusion
from mmseg.ops import resize


# from .dfm import DF_Module


class ConvModule(nn.Module):
    def __init__(self, filters, kernel_size: int = 3, norm_cfg=True, act_cfg=True):
        super().__init__()
        layers = []
        for i in range(1, len(filters)):
            layers.extend([
                nn.Conv2d(filters[i - 1], filters[i], kernel_size, padding=kernel_size // 2),
                nn.BatchNorm2d(filters[i]) if norm_cfg else nn.Identity(),
                nn.ReLU(inplace=True) if act_cfg else nn.Identity()
            ])
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        return x


class ASPP3d(nn.ModuleList):

    def __init__(self, features, out_features=None, dilations=(1, 2, 4, 8)):
        super(ASPP3d, self).__init__()

        self.out_features = out_features if out_features else features
        self.dilations = dilations

        for dilation in dilations:
            if dilation == 1:
                kernel_size = (2, 1, 1)
                padding = (0, 0, 0)
                dilation = (1,) + to_2tuple(dilation)
                self.append(
                    nn.Sequential(
                        nn.Conv3d(features, self.out_features,
                                  kernel_size=kernel_size,
                                  padding=padding,
                                  dilation=dilation,
                                  bias=False),
                        nn.BatchNorm3d(self.out_features),
                        nn.ReLU(inplace=True)
                    )
                )
            else:
                kernel_size = (2, 3, 3)
                self.append(
                    nn.Sequential(
                        nn.Conv3d(features, self.out_features,
                                  kernel_size=kernel_size,
                                  padding=(0,) + to_2tuple(dilation),
                                  dilation=(1,) + to_2tuple(dilation),
                                  bias=False),
                        nn.BatchNorm3d(self.out_features),
                        nn.ReLU(inplace=True)
                    )
                )

    def forward(self, x):
        """Forward function."""
        # fusion_feas = torch.stack([A, B], dim=2)

        h, w = x.size()[-2:]
        # print(x.shape)
        aspp_outs = []
        for idx, aspp_module in enumerate(self):
            aspp_outs.append(aspp_module(x).squeeze(2)) if idx > 0 else aspp_outs.append(
                resize(aspp_module(x).squeeze(2), size=(h, w)))

        return aspp_outs


class AsppFusion(nn.Module):
    def __init__(self, features, out_features=None, dilations=(1, 8, 12, 24), act_cfg=True, norm_cfg=True, ):
        super().__init__()

        self.ppm = ASPP3d(features=features, out_features=out_features,
                          dilations=dilations)

        self.bottleneck = ConvModule(
            [len(dilations) * self.ppm.out_features, self.ppm.out_features],
            3,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg) if len(dilations) > 1 else nn.Identity()

    # def forward(self, x, y):
    def forward(self, fusion_fea):
        # fusion_fea = torch.stack([x, y], dim=2)
        fusion_fea = torch.cat(self.ppm(fusion_fea), dim=1)
        fusion_fea = self.bottleneck(fusion_fea)

        return fusion_fea


class Aspp3dFusion(nn.Module):
    def __init__(self, in_channels, out_features=None, dilations=(1, 8, 12, 24), act_cfg=True, norm_cfg=True, **kwargs):
        assert isinstance(in_channels, (list, tuple))

        super().__init__()
        if out_features is None:
            out_features = in_channels

        self.ppms = nn.ModuleList([AsppFusion(
            features=channels, dilations=dilations, out_features=out_channels, act_cfg=act_cfg, norm_cfg=norm_cfg) for
            channels, out_channels in
            zip(in_channels, out_features)])

    def forward(self, x1, x2):
        """[summary]
        将 （b c h w）x2 ->  b c h w

        Args:
            x1 ([type]): [（b c h w）]
            x2 ([type]): [（b c h w）]
            output :（b 2c  h w）
        """
        # assert isinstance(x1, (list, tuple))
        # assert isinstance(x2, (list, tuple))
        if not isinstance(x1, (list, tuple)):
            x1 = [x1]
            x2 = [x2]

        out = []
        for x, y, ppm in zip(x1, x2, self.ppms):
            out.append(ppm(x, y))
        return out


# ----------------------------
class Sum(BaseModule):
    def __init__(self, in_channels, out_channels, act_cfg, norm_cfg, last_conv=False, kernel_size=3, **kwargs):
        super(Sum, self).__init__()

        self.last_conv = nn.ModuleList()

        if out_channels is None:
            out_channels = in_channels

        if last_conv:
            for in_c, out_c in zip(in_channels, out_channels):
                self.last_conv.append(MMConvModule(
                    in_channels=in_c,
                    out_channels=out_c,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                ))
        else:
            for in_c, out_c in zip(in_channels, out_channels):
                self.last_conv.append(nn.Identity())

    def forward(self, x1, x2):
        if not isinstance(x1, (list, tuple)):
            x1 = [x1]
            x2 = [x2]

        out = []
        for x, y, ppm in zip(x1, x2, self.last_conv):
            out.append(ppm(x + y))
        return out


class DF(BaseModule):
    def __init__(self, in_channels, out_channels, reduction=False, **kwargs):
        super(DF, self).__init__()
        print(reduction)

        if out_channels is None:
            out_channels = in_channels

        self.dense_cat = nn.ModuleList()

        for in_c, out_c in zip(in_channels, out_channels):
            self.dense_cat.append(DF_Module(dim_in=in_c, dim_out=out_c, reduction=reduction))

    def forward(self, x1, x2):
        if not isinstance(x1, (list, tuple)):
            x1 = [x1]
            x2 = [x2]

        out = []
        for x, y, ppm in zip(x1, x2, self.dense_cat):
            out.append(ppm(x, y))
        return out


class Diff(BaseModule):

    def __init__(self, in_channels, out_channels, act_cfg, norm_cfg, last_conv=False, kernel_size=3, **kwargs):
        super(Diff, self).__init__()

        self.last_conv = nn.ModuleList()

        if out_channels is None:
            out_channels = in_channels

        if last_conv:
            for in_c, out_c in zip(in_channels, out_channels):
                self.last_conv.append(MMConvModule(
                    in_channels=in_c,
                    out_channels=out_c,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    order=('act', 'conv', 'norm')
                ))
        else:
            for in_c, out_c in zip(in_channels, out_channels):
                self.last_conv.append(nn.Identity())

    def forward(self, x1, x2):
        if not isinstance(x1, (list, tuple)):
            x1 = [x1]
            x2 = [x2]

        out = []
        for x, y, ppm in zip(x1, x2, self.last_conv):
            out.append(ppm(torch.abs(x - y)))
        return out


class Concate(BaseModule):
    def __init__(self, in_channels, out_channels, act_cfg, norm_cfg, last_conv=False, kernel_size=3, **kwargs):
        super(Concate, self).__init__()

        self.last_conv = nn.ModuleList()

        if out_channels is None:
            out_channels = in_channels

        if last_conv:
            for in_c, out_c in zip(in_channels, out_channels):
                self.last_conv.append(MMConvModule(
                    in_channels=in_c * 2,
                    out_channels=out_c,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                ))
        else:
            for in_c, out_c in zip(in_channels, out_channels):
                self.last_conv.append(nn.Identity())

    def forward(self, x1, x2):
        if not isinstance(x1, (list, tuple)):
            x1 = [x1]
            x2 = [x2]

        out = []
        for x, y, ppm in zip(x1, x2, self.last_conv):
            out.append(ppm(torch.cat([x, y], dim=1)))
        return out


class Concate3d(BaseModule):
    def __init__(self, in_channels, out_channels, act_cfg, norm_cfg, kernel_size=1, **kwargs):
        super(Concate3d, self).__init__()

        self.last_conv = nn.ModuleList()

        if out_channels is None:
            out_channels = in_channels

        for in_c, out_c in zip(in_channels, out_channels):
            self.last_conv.append(MMConvModule(
                in_channels=in_c,
                out_channels=out_c,
                conv_cfg=dict(type='Conv3d'),
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                kernel_size=(2,) + to_2tuple(kernel_size),
                padding=(0,) + to_2tuple(kernel_size // 2),
            ))

    def forward(self, x1, x2):
        if not isinstance(x1, (list, tuple)):
            x1 = [x1]
            x2 = [x2]

        out = []
        for x, y, ppm in zip(x1, x2, self.last_conv):
            out.append(ppm(torch.stack([x, y], dim=2)).squeeze(2))
        return out


from ..builder import CHANGES, build_plugin


# ------------------------------------
@CHANGES.register_module()
class FusionHEAD(BaseModule, metaclass=ABCMeta):

    def __init__(self,
                 in_channels: Union[List, int],
                 channels: Union[List, int],
                 fusion_form: str = "abs_diff",
                 act_cfg: Union[Dict, None] = dict(type='ReLU'),
                 norm_cfg: Union[Dict, None] = None,
                 init_cfg: Dict = dict(
                     type='Normal', std=0.01),
                 **kwargs
                 ):
        """

        :param in_channels: 输入通道数，可以是list，也可以是int
        :param channels:  输出通道数，可以是list，也可以是int
        :param fusion_form: 融合类型
        :param act_cfg:
        :param norm_cfg:
        :param init_cfg:
        :param kwargs: 内部参数  last_conv=True, kernel_size=3, dilations=(1, 8, 12, 24)
        """
        super(FusionHEAD, self).__init__(init_cfg)

        fusion_form = fusion_form.lower()

        assert fusion_form in ['abs_diff', 'sum', 'concate', 'aspp3d', 'dense_cat', 'concate3d']

        if isinstance(in_channels, list):
            self.list_out = True
        else:
            self.list_out = False
            in_channels = [in_channels]
            channels = [channels]

        if fusion_form == 'abs_diff':
            self.FuseLayer = Diff(in_channels=in_channels, out_channels=channels, act_cfg=act_cfg,
                                  norm_cfg=norm_cfg, **kwargs)
        elif fusion_form == 'sum':
            self.FuseLayer = Sum(in_channels=in_channels, out_channels=channels, act_cfg=act_cfg,
                                 norm_cfg=norm_cfg, **kwargs)

        elif fusion_form == 'concate':
            self.FuseLayer = Concate(in_channels=in_channels, out_channels=channels, act_cfg=act_cfg,
                                     norm_cfg=norm_cfg, **kwargs)

        elif fusion_form == 'dense_cat':
            self.FuseLayer = DF(in_channels, channels, **kwargs)

        elif fusion_form == 'aspp3d':
            self.FuseLayer = Aspp3dFusion(in_channels, channels, act_cfg=act_cfg, norm_cfg=norm_cfg, **kwargs)

        elif fusion_form == 'concate3d':
            if norm_cfg:
                assert norm_cfg['type'] == "BN3d"
            self.FuseLayer = Concate3d(in_channels, channels, act_cfg=act_cfg, norm_cfg=norm_cfg, **kwargs)

        else:
            raise NotImplementedError("{} is not implemented".format(fusion_form))

    def forward(self, x1, x2):
        if self.list_out:
            return self.FuseLayer(x1, x2)
        else:
            return self.FuseLayer(x1, x2)[0]


if __name__ == '__main__':
    import torch


    def cal_params(model):
        import numpy as np
        p = filter(lambda p: p.requires_grad, model.parameters())
        p = sum([np.prod(p_.size()) for p_ in p]) / 1_000_000
        print('%.3fM' % p)


    in_channels = [128, 256, 512, 1024]
    scales = [340, 170, 84, 43]
    a = [torch.rand(1, c, s, s) for c, s in zip(in_channels, scales)]
    b = [torch.rand(1, c, s, s) for c, s in zip(in_channels, scales)]

    # fuse = FusionHEAD(in_channels=2, channels=4, fusion_form='concate', act_cfg=None, norm_cfg=None, dilations=[1, 4])
    fuse = FusionHEAD(in_channels=in_channels, channels=[128, 256, 512, 1024], fusion_form='dense_cat', last_conv=True,
                      norm_cfg=dict(type="BN3d"), reduction=True,
                      kernel_size=1)

    cal_params(fuse)
    out = fuse(a, b)

    if isinstance(out, list):
        for o in out:
            print(o.shape)
    else:
        print(out.shape)
