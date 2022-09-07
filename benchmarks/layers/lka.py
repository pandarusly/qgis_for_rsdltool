import torch
import torch.nn as nn
from torch.nn.modules.conv import Conv2d

from benchmarks._builder import CHANGES


@CHANGES.register_module()
class LKADIFF(nn.Module):
    def __init__(self, dim, k, td=True, k2=3) -> None:
        super().__init__()
        pad = k // 2
        self.td = td
        if td:
            self.conv0 = nn.Conv3d(
                dim, dim,
                kernel_size=(2, k, k),
                padding=(0, pad, pad),
                groups=dim
            )
        else:
            self.conv0 = nn.Conv2d(
                dim, dim,
                kernel_size=k,
                padding=pad,
                groups=dim
            )

        self.point_wise = Conv2d(
            dim, dim, kernel_size=1
        )

        if k2 != 0:
            self.conv1 = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=k2, padding=k2 // 2),
                nn.ReLU()
            )
        else:
            self.conv1 = nn.Identity()

    def forward_ed(self, x, y):
        u = x.clone()
        x = self.conv0(x)
        x = self.point_wise(x)
        u = u * x
        u = self.conv1(u)

        v = y.clone()
        y = self.conv0(y)
        y = self.point_wise(y)
        v = v * y
        v = self.conv1(v)
        return u, v

    def forward_td(self, x, y):
        u = x.clone()
        v = y.clone()
        attn = torch.stack([x, y], dim=2)  # b c t h w
        attn = self.conv0(attn).squeeze(2)  # b c h w
        attn = self.point_wise(attn)
        u = u * attn
        u = self.conv1(u)
        v = v * attn
        v = self.conv1(v)
        return u, v

    def forward(self, x, y):
        if self.td:
            x, y = self.forward_td(x, y)
        else:
            x, y = self.forward_ed(x, y)

        return torch.abs(x - y)


@CHANGES.register_module()
class LKADIFFS(nn.Module):
    def __init__(self, dim, k, k2=3, **kwargs) -> None:
        super().__init__()
        isinstance(k, (list, tuple))

        self.conv0 = nn.ModuleList()

        for k_ in k:
            pad = k_ // 2
            self.conv0.append(nn.Conv3d(dim, dim, kernel_size=(2, k_, k_), padding=(0, pad, pad), groups=dim))

        self.point_wise = Conv2d(
            dim * len(k), dim, kernel_size=1
        )

        if k2 != 0:  # 并不会带来更多的提升，只不过变化特征图更加清晰了 (可选)
            self.conv1 = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=k2, padding=k2 // 2),
                nn.ReLU()
            )
        else:
            self.conv1 = nn.Identity()

    def forward_td(self, x, y):
        u = x.clone()
        v = y.clone()
        t1t2 = torch.stack([x, y], dim=2)  # b c t h w
        attn = []
        for dw_conv in self.conv0:
            attn.append(dw_conv(t1t2).squeeze(2))  # b c h w
        attn = torch.cat(attn, dim=1)
        attn = self.point_wise(attn)
        u = u * attn
        u = self.conv1(u)
        v = v * attn
        v = self.conv1(v)
        return u, v

    def forward(self, x, y):

        x, y = self.forward_td(x, y)

        return torch.abs(x - y)


class Squeeze3d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.squeeze(2)


@CHANGES.register_module()
class LKADIFFSV2(nn.Module):
    def __init__(self, dim, k, k2=3, **kwargs) -> None:
        super().__init__()
        isinstance(k, (list, tuple))

        self.conv0 = nn.ModuleList()

        for k_ in k:
            pad = k_ // 2
            self.conv0.append(
                nn.Sequential(
                    nn.Conv3d(dim, dim, kernel_size=(2, k_, k_), padding=(0, pad, pad), groups=dim),
                    Squeeze3d(),
                    nn.Conv2d(dim, dim, kernel_size=1)
                )
            )

        self.point_wise = Conv2d(
            dim * len(k), dim, kernel_size=1
        )

        if k2 != 0:  # 并不会带来更多的提升，只不过变化特征图更加清晰了 (可选)
            self.conv1 = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=k2, padding=k2 // 2),
                nn.ReLU()
            )
        else:
            self.conv1 = nn.Identity()

    def forward_td(self, x, y):
        u = x.clone()
        v = y.clone()
        t1t2 = torch.stack([x, y], dim=2)  # b c t h w
        attn = []
        for dw_conv in self.conv0:
            attn.append(dw_conv(t1t2))  # b c h w
        attn = torch.cat(attn, dim=1)
        attn = self.point_wise(attn)
        u = u * attn
        u = self.conv1(u)
        v = v * attn
        v = self.conv1(v)
        return u, v

    def forward(self, x, y):

        x, y = self.forward_td(x, y)

        return torch.abs(x - y)


if __name__ == '__main__':
    import torch


    def cal_params(model):
        import numpy as np
        p = filter(lambda p: p.requires_grad, model.parameters())
        p = sum([np.prod(p_.size()) for p_ in p]) / 1_000_000
        print('%.3fM' % p)


    in_channels = [32, 64, 160, 256]
    # [128//2, 256//2, 512//2, 1024//2]
    scales = [340, 170, 84, 43]
    in_channels = [128]
    scales = [256]
    a = [torch.rand(1, c, s, s) for c, s in zip(in_channels, scales)]
    b = [torch.rand(1, c, s, s) for c, s in zip(in_channels, scales)]

    # fuse = FusionHEAD(in_channels=2, channels=4, fusion_form='attn', act_cfg=None, norm_cfg=None, dilations=[1, 4])
    fuse = LKADIFFS(128, [13, 3, 5, 7])

    cal_params(fuse)
    x = fuse(a[0], b[0])
    print(x.shape)
    fuse = LKADIFF(128, 13, True)
    cal_params(fuse)
    x = fuse(a[0], b[0])
    print(x.shape)
    fuse = LKADIFFSV2(128, [13, 3, 5, 7])
    cal_params(fuse)
    x = fuse(a[0], b[0])
    print(x.shape)
