from benchmarks.backbones.resnetv2 import build_backbone
from benchmarks.MixOC import BACKBONES
from trainers.utils.weight_init import init_weights
from ._builder import BENCHMARKS
import torch
import torch.nn as nn
import torch.nn.functional as F


class DR(nn.Module):
    def __init__(self, in_d, out_d):
        super(DR, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.conv1 = nn.Conv2d(self.in_d, self.out_d, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_d)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        return x


class Decoder(nn.Module):
    def __init__(self, fc, BatchNorm):
        super(Decoder, self).__init__()
        self.fc = fc
        self.dr2 = DR(64, 96)
        self.dr3 = DR(128, 96)
        self.dr4 = DR(256, 96)
        self.dr5 = DR(512, 96)
        self.last_conv = nn.Sequential(nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(
                                           256, self.fc, kernel_size=1, stride=1, padding=0, bias=False),
                                       BatchNorm(self.fc),
                                       nn.ReLU(),
                                       )
        self._init_weight()

    def forward(self, x, low_level_feat2, low_level_feat3, low_level_feat4):

        x2 = self.dr2(low_level_feat2)
        x3 = self.dr3(low_level_feat3)
        x4 = self.dr4(low_level_feat4)
        x = self.dr5(x)

        x = F.interpolate(x, size=x2.size()[
                          2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=x2.size()[
                           2:], mode='bilinear', align_corners=True)
        x4 = F.interpolate(x4, size=x2.size()[
                           2:], mode='bilinear', align_corners=True)

        x = torch.cat((x, x2, x3, x4), dim=1)

        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_decoder(fc, BatchNorm):
    return Decoder(fc, BatchNorm)


# ----------------------注意力
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


class DS_layer(nn.Module):
    def __init__(self, in_d, out_d, stride, output_padding, n_class):
        super(DS_layer, self).__init__()

        self.dsconv = nn.ConvTranspose2d(in_d, out_d, kernel_size=3, padding=1, stride=stride,
                                         output_padding=output_padding)
        self.bn = nn.BatchNorm2d(out_d)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=0.2)
        self.outconv = nn.ConvTranspose2d(
            out_d, n_class, kernel_size=3, padding=1)

    def forward(self, input):
        x = self.dsconv(input)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.outconv(x)
        return x


def torch10_pairwise_distance(x1, x2):
    x1 = x1.permute(0, 2, 3, 1)
    x2 = x2.permute(0, 2, 3, 1)
    x = F.pairwise_distance(x1, x2, keepdim=True)
    x = x.permute(0, 3, 1, 2)
    return x


@BENCHMARKS.register_module()
class DSAMNet(nn.Module):
    def __init__(self, n_class=2,  ratio=8, kernel=7, backbone='resnet18', output_stride=16, f_c=64, freeze_bn=False, in_c=3, up_torchversion=True):
        super(DSAMNet, self).__init__()
        BatchNorm = nn.BatchNorm2d
        self.up_torchversion = up_torchversion

        self.backbone = build_backbone(
            backbone, output_stride, BatchNorm, in_c)
        self.decoder = build_decoder(f_c, BatchNorm)

        self.cbam0 = CBAM(f_c, ratio, kernel)
        self.cbam1 = CBAM(f_c, ratio, kernel)

        self.ds_lyr2 = DS_layer(64, 32, 2, 1, n_class)
        self.ds_lyr3 = DS_layer(128, 32, 4, 3, n_class)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, img1, img2):
        x_1, f2_1, f3_1, f4_1 = self.backbone(img1)
        x_2, f2_2, f3_2, f4_2 = self.backbone(img2)

        x1 = self.decoder(x_1, f2_1, f3_1, f4_1)
        x2 = self.decoder(x_2, f2_2, f3_2, f4_2)

        x1 = self.cbam0(x1)
        x2 = self.cbam1(x2)  # channel = 64

        if self.up_torchversion:
            dist = torch10_pairwise_distance(x1, x2)
        else:
            dist = F.pairwise_distance(x1, x2, keepdim=True)  # channel = 1
        dist = F.interpolate(
            dist, size=img1.shape[2:], mode='bilinear', align_corners=True)

        ds2 = self.ds_lyr2(torch.abs(f2_1 - f2_2))
        ds3 = self.ds_lyr3(torch.abs(f3_1 - f3_2))

        return dist, ds2, ds3

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def init_weights(self):
        init_weights(self.decoder)
        init_weights(self.cbam0)
        init_weights(self.cbam1)
        init_weights(self.ds_lyr2)
        init_weights(self.ds_lyr3)
