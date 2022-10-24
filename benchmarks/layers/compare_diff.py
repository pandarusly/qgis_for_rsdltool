import torch.nn as nn
from timm.models.layers import create_attn

from benchmarks._builder import CHANGES


@CHANGES.register_module()
class CompareDIFFSE(nn.Module):
    def __init__(self, dim, k=1, td=True) -> None:
        super().__init__()
        pad = k // 2
        self.td = td

        if td:
            self.conv0 = nn.Sequential(
                create_attn('se', dim * 2),
                nn.Conv2d(
                    dim * 2, dim,
                    kernel_size=k,
                    padding=pad
                )
            )
        else:
            self.conv0 = nn.Identity()

        self.point_wise = nn.Identity()

        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward_td(self, x, y):
        u = x.clone()
        v = y.clone()
        attn = torch.cat([x, y], dim=1)  # b 2c h w
        attn = self.conv0(attn)  # b c h w
        attn = self.point_wise(attn)
        u = u * attn
        u = self.conv1(u)
        v = v * attn
        v = self.conv1(v)
        return u, v

    def forward_ed(self, x, y):
        x = self.conv1(x)
        y = self.conv1(y)
        return x, y

    def forward(self, x, y):
        if self.td:
            x, y = self.forward_td(x, y)
        else:
            x, y = self.forward_ed(x, y)
        return torch.abs(x - y)


@CHANGES.register_module()
class CompareDIFF(nn.Module):
    def __init__(self, dim, k=1, td=True, k2=0, **kwargs) -> None:
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
            self.point_wise = nn.Conv2d(
                dim, dim, kernel_size=1
            )
        else:
            self.conv0 = nn.Identity()

            self.point_wise = nn.Identity()

        if k2 == 0:
            self.conv1 = nn.Identity()
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=k2, padding=k2 // 2),
                nn.ReLU()
            )

    def forward_td(self, x, y):
        attn = torch.stack([x, y], dim=2)  # b c t h w
        attn = self.conv0(attn).squeeze(2)  # b c h w
        attn = self.point_wise(attn)
        attn = self.conv1(attn)
        return attn

    def forward_ed(self, x, y):
        x = self.conv1(x)
        y = self.conv1(y)
        return torch.abs(x - y)

    def forward(self, x, y):
        if self.td:
            diff = self.forward_td(x, y)
        else:
            diff = self.forward_ed(x, y)
        return diff


@CHANGES.register_module()
class Concate(nn.Module):
    def __init__(self, dim, k=3, **kwargs) -> None:
        super().__init__()
        pad = k // 2
        self.conv0 = nn.Conv3d(
            dim, dim,
            kernel_size=(2, k, k),
            padding=(0, pad, pad),
        )

    def forward(self, x, y):
        diff = torch.stack([x, y], dim=2)  # b c t h w
        diff = self.conv0(attn).squeeze(2)  # b c h w
        return attn


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

    k = 3
    k2 = 3
    cc = 2048
    in_channels = [cc]
    scales = [8]
    a = [torch.rand(1, c, s, s) for c, s in zip(in_channels, scales)]
    b = [torch.rand(1, c, s, s) for c, s in zip(in_channels, scales)]

    # fuse = FusionHEAD(in_channels=2, channels=4, fusion_form='attn', act_cfg=None, norm_cfg=None, dilations=[1, 4])
    fuse = CompareDIFF(cc, k, True, k2)
    comp_fuse = nn.Conv2d(cc * 2, cc, kernel_size=k)
    cal_params(comp_fuse)
    cal_params(fuse)
    x = fuse(a[0], b[0])
    print(x.shape)
    fuse = CompareDIFF(cc, k, False, k2)
    cal_params(fuse)
    x = fuse(a[0], b[0])
    print(x.shape)
