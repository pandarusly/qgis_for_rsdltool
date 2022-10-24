# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from benchmarks._builder import CHANGES


# -----------------dense_block_3d
class dense_block_3d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2 + x1)

        return x1 + x2 + x3


@CHANGES.register_module()
class attn_fusion_cp(nn.Module):
    def __init__(self, in_chn=128, out_chn=128, r=8):
        super().__init__()

        self.attn = nn.Sequential(
            dense_block_3d(out_chn),  # 3
            Rearrange("b c t h w -> b (c t) h w"),  # 4
            nn.Conv2d(in_chn * 2, out_chn, kernel_size=1, padding=0),  # 5
            nn.BatchNorm2d(out_chn),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        x_3d = torch.stack([x, y], dim=2)
        return self.attn(x_3d)


# ------------------- dense_diff
@CHANGES.register_module()
class densecat_cat_diff_cp(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(densecat_cat_diff_cp, self).__init__()

        if in_chn != out_chn:
            self.reduction = torch.nn.Sequential(
                torch.nn.Conv2d(in_chn, out_chn, kernel_size=1, padding=0),
                nn.BatchNorm2d(out_chn),
                torch.nn.ReLU(inplace=True),
            )
        else:
            self.reduction = torch.nn.Identity()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(out_chn, out_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(out_chn, out_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(out_chn, out_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv2d(out_chn, in_chn, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_chn),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        x = self.reduction(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2 + x1)

        y = self.reduction(y)
        y1 = self.conv1(y)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2 + y1)
        out = self.conv_out(torch.abs(x1 + x2 + x3 - y1 - y2 - y3))
        return out


# ------------------- dw_diff
from timm.models.layers import DropPath


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        kernel_size (int):  e. Default: 7.
    """

    def __init__(self, dim, kernel_size=7, scale=0.5):
        super().__init__()
        padding = kernel_size // 2
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size,
                                padding=padding, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        # pointwise/1x1 convs, implemented with linear layers
        self.pwconv1 = nn.Linear(dim, int(scale * dim))
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(int(scale * dim), dim)

    def forward(self, x):
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return x


class ResBlock(nn.Module):
    def __init__(self, dim, drop_path=0., kernel_size=7, scale=0.5):
        super().__init__()
        padding = kernel_size // 2
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size,
                                padding=padding, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        # pointwise/1x1 convs, implemented with linear layers
        self.pwconv1 = nn.Linear(dim, int(scale * dim))
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(int(scale * dim), dim)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x


@CHANGES.register_module()
class depwise_diff(nn.Module):
    def __init__(self, in_chn=128, out_chn=128, scale=0.5, kernel_size=7, drop_path_rate=0.0, depth=0):

        super().__init__()

        if in_chn != out_chn:
            self.reduction = torch.nn.Sequential(
                torch.nn.Conv2d(in_chn, out_chn, kernel_size=1, padding=0),
                nn.BatchNorm2d(out_chn),
                torch.nn.ReLU(inplace=True),
            )
        else:
            self.reduction = torch.nn.Identity()

        convs = []
        # drop_path_rate = drop_path_rate
        convs.append(
            nn.Sequential(
                Block(out_chn, kernel_size=kernel_size, scale=scale)
            )
        )

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, depth - 1)] if depth != 0 else [0.0]

        for i in range(depth - 1):
            convs.append(
                nn.Sequential(
                    ResBlock(out_chn, kernel_size=kernel_size, drop_path=dp_rates[i], scale=scale)
                )
            )

        if depth == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)

        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv2d(out_chn, in_chn, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_chn),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        x = self.reduction(x)
        x1 = self.convs(x)

        y = self.reduction(y)
        y1 = self.convs(y)
        out = self.conv_out(torch.abs(x1 - y1))
        return out
