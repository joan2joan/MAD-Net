import torch
import torch.nn as nn
from einops import rearrange

from model.MAD.Module import conv_block, up_conv, _upsample_like, MemA2M, AdaDCM


class MADNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, dim=64, ori_h=256, deep_supervision=True, **kwargs):
        super(MADNet, self).__init__()
        self.deep_supervision = deep_supervision
        filters = [dim, dim * 2, dim * 4, dim * 8, dim * 16]
        self.maxpools = nn.ModuleList([nn.MaxPool2d(kernel_size=2, stride=2) for _ in range(4)])
        self.Conv1 = AdaDCM(in_ch=in_ch, out_ch=filters[0])
        self.Convtans2 =AdaDCM(in_ch=filters[0], out_ch=filters[1])
        self.Convtans3 =AdaDCM(in_ch=filters[1], out_ch=filters[2])
        self.Convtans4 =AdaDCM(in_ch=filters[2], out_ch=filters[3])
        self.Conv5 = MemA2M(in_ch=filters[3], out_ch=filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = MemA2M(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        # --------------------------------------------------------------------------------------------------------------
        self.conv5 = nn.Conv2d(filters[4], out_ch, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(filters[3], out_ch, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(filters[2], out_ch, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(filters[1], out_ch, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(filters[0], out_ch, kernel_size=3, stride=1, padding=1)
        # --------------------------------------------------------------------------------------------------------------

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.maxpools[0](e1)
        e2 = self.Convtans2(e2)

        e3 = self.maxpools[1](e2)
        e3 = self.Convtans3(e3)

        e4 = self.maxpools[2](e3)
        e4 = self.Convtans4(e4)

        e5 = self.maxpools[3](e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        d_s1 = self.conv1(d2)
        d_s2 = self.conv2(d3)
        d_s2 = _upsample_like(d_s2, d_s1)
        d_s3 = self.conv3(d4)
        d_s3 = _upsample_like(d_s3, d_s1)
        d_s4 = self.conv4(d5)
        d_s4 = _upsample_like(d_s4, d_s1)
        d_s5 = self.conv5(e5)
        d_s5 = _upsample_like(d_s5, d_s1)
        if self.deep_supervision:
            outs = [d_s1, d_s2, d_s3, d_s4, d_s5, out]
        else:
            outs = out

        return outs


if __name__ == '__main__':
    x = torch.randn(8, 3, 256, 256)
    model = ABCNet(ori_h=256)
    y = model(x)
    