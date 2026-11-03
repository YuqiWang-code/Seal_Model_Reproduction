import torch
import torch.nn as nn
import torch.nn.functional as F


class CDC_conv(nn.Module):
    """
    Central Difference Convolution (CDC) layer.
    This module implements the Central Difference Convolution as described in the paper:
    "Central Difference Convolutional Networks" (https://arxiv.org/abs/2003.04054).
    """

    def __init__(self, in_channels, out_channels, bias=True, kernel_size=3, padding=1, dilation=1, theta=0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                              bias=bias)
        self.theta = theta

    def forward(self, x):
        norm_out = self.conv(x)
        kernel_diff = self.conv.weight.sum(2).sum(2)
        kernel_diff = kernel_diff[:, :, None, None]
        diff_out = F.conv2d(input=x, weight=kernel_diff,
                            bias=self.conv.bias, stride=self.conv.stride, padding=0)
        out = norm_out - self.theta * diff_out
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        norm = nn.BatchNorm2d
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1,
                      bias=False if norm == nn.BatchNorm2d else True),
            norm(out_c),
            nn.ReLU(inplace=True),
            CDC_conv(out_c, out_c, kernel_size=3, padding=1,
                     bias=False if norm == nn.BatchNorm2d else True),
            norm(out_c),
        )
        self.residual_block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv_out = self.conv_block(x)
        residual_out = self.residual_block(x)
        out = self.relu(conv_out + residual_out)
        return out


class Encoder(nn.Module):
    """
    Encoder backbone that extracts multi-scale feature maps using an initial
    convolutional stem followed by three residual stages with progressive
    downsampling.
    """

    def __init__(self, in_channel=3, filters=None):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.initial_conv = nn.Sequential(
            CDC_conv(in_channel, filters[0], bias=False),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True),
            CDC_conv(filters[0], filters[0], bias=False),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ResidualBlock(filters[0], filters[1])
        self.layer2 = ResidualBlock(filters[1], filters[2])
        self.layer3 = ResidualBlock(filters[2], filters[3])

    def forward(self, x):
        out_0 = self.initial_conv(x)
        out_1 = self.layer1(self.maxpool(out_0))
        out_2 = self.layer2(self.maxpool(out_1))
        out_3 = self.layer3(self.maxpool(out_2))
        return out_0, out_1, out_2, out_3
