from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch

from model.MAD.mema2m import mema2m
from model.MAD.diffconv import DEConv


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


def _upsample_like(src, tar):
    src = F.upsample(src, size=tar.shape[2:], mode='bilinear')
    return src


class MemA2M(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MemA2M, self).__init__()
        self.conv1 = conv_relu_bn(in_ch, out_ch, 1)
        drop_path_rate=0.1
        depth = [4]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        self.mema2m = nn.Sequential(*[mema2m(dim=out_ch, input_resolution=16, num_heads=4, memory_blocks=128,
                                             window_size=16,shift_size=0 if i%2 == 0 else 8, weight_factor=0.1,
                                             down_rank=8, mlp_ratio=2, drop_path=dpr[i],)
                                    for i in range(depth[0])])
    def forward(self, x):
        x1 = self.conv1(x) 
        out = self.mema2m(x1)
        return out


class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        pattn2 = self.pa2(x)
        pattn2 = self.sigmoid(pattn2)
        return pattn2


class AdaDCM(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(AdaDCM, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True))
        self.conv1 = DEConv(out_ch)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=1, bias=True)
        self.pa = PixelAttention(out_ch)

    def forward(self, x):
        x = self.conv(x)
        res = self.conv1(x)
        res = self.act1(res)
        res = res + x
        res = self.conv2(res)
        pattn2 = self.pa(res)
        res = res * pattn2
        res = res + x
        return res