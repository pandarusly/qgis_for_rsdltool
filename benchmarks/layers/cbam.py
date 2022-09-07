import torch.nn as nn

from benchmarks._builder import CHANGES


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio, kernel_size):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


@CHANGES.register_module()
class CBAMDIFF(nn.Module):
    def __init__(self, dim, k=None, td=False, k2=0) -> None:
        super().__init__()

        self.conv0 = CBAM(dim, 8, 7)

        self.point_wise = nn.Identity()

        k2 = 0
        if k2 != 0:
            self.conv1 = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=k2, padding=k2 // 2),
                nn.ReLU()
            )
        else:
            self.conv1 = nn.Identity()

    def forward_ed(self, x, y):

        x = self.conv0(x)
        x = self.point_wise(x)
        y = self.conv0(y)
        y = self.point_wise(y)

        return x, y

    def forward(self, x, y):

        x, y = self.forward_ed(x, y)

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

    fuse = CBAMDIFF(128, 13, True)
    cal_params(fuse)
    x = fuse(a[0], b[0])
    print(x.shape)
