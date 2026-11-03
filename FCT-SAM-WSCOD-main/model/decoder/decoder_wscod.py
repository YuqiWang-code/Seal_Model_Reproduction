import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from einops import rearrange


def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Sequential, nn.ModuleList)):
            weight_init(m)
        elif isinstance(m, (
        nn.ReLU, nn.GELU, nn.ReLU6, nn.Sigmoid, nn.Softmax, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool1d, nn.UpsamplingBilinear2d,
        nn.Sigmoid, nn.Identity)):
            pass
        else:
            m.initialize()


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class PyramidPooling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PyramidPooling, self).__init__()
        hidden_channel = int(in_channel / 4)
        self.conv1 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv2 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv3 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv4 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.out = basicConv(in_channel * 2, out_channel, k=1, s=1, p=0)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = F.interpolate(self.conv1(F.adaptive_avg_pool2d(x, 1)), size)
        feat2 = F.interpolate(self.conv2(F.adaptive_avg_pool2d(x, 2)), size)
        feat3 = F.interpolate(self.conv3(F.adaptive_avg_pool2d(x, 3)), size)
        feat4 = F.interpolate(self.conv4(F.adaptive_avg_pool2d(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)

        return x

    def initialize(self):
        weight_init(self)

class basicConv(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(basicConv, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
            # conv.append(nn.LayerNorm(out_channel, eps=1e-6))
        if relu:
            conv.append(nn.GELU())
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)

    def initialize(self):
        weight_init(self)


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

    def initialize(self):
        weight_init(self)


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

    def initialize(self):
        weight_init(self)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

    def initialize(self):
        weight_init(self)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, mode):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv_0 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.qkv1conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv2conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv3conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, mask=None):
        b, c, h, w = x.shape
        q = self.qkv1conv(self.qkv_0(x))
        k = self.qkv2conv(self.qkv_1(x))
        v = self.qkv3conv(self.qkv_2(x))
        if mask is not None:
            q = q * mask
            k = k * mask

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

    def initialize(self):
        weight_init(self)


class MSA_head(nn.Module):
    def __init__(self, mode='dilation', dim=64, num_heads=8, ffn_expansion_factor=4, bias=False,
                 LayerNorm_type='WithBias'):
        super(MSA_head, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias, mode)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x

    def initialize(self):
        weight_init(self)

# Revised from: PraNet: Parallel Reverse Attention Network for Polyp Segmentation, MICCAI20
# https://github.com/DengPingFan/PraNet
class RFB_modified(nn.Module):
    """ logical semantic relation (LSR) """

    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            basicConv(in_channel, out_channel, 1, relu=False),
        )
        self.branch1 = nn.Sequential(
            basicConv(in_channel, out_channel, 1),
            basicConv(out_channel, out_channel, k=7, p=3),
            basicConv(out_channel, out_channel, 3, p=7, d=7, relu=False)
        )
        self.branch2 = nn.Sequential(
            basicConv(in_channel, out_channel, 1),
            basicConv(out_channel, out_channel, k=7, p=3),
            basicConv(out_channel, out_channel, k=7, p=3),
            basicConv(out_channel, out_channel, 3, p=7, d=7, relu=False)
        )
        self.branch3 = nn.Sequential(
            basicConv(in_channel, out_channel, 1),
            basicConv(out_channel, out_channel, k=7, p=3),
            basicConv(out_channel, out_channel, k=7, p=3),
            basicConv(out_channel, out_channel, 3, p=7, d=7, relu=False)
        )
        self.conv_cat = basicConv(4 * out_channel, out_channel, 3, p=1, relu=False)
        self.conv_res = basicConv(in_channel, out_channel, 1, relu=False)

    def forward(self, x):

        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x_temp = self.relu(x_cat + self.conv_res(x))
        x = F.interpolate(x_temp, size=x.size()[2:], mode='bilinear')
        return x

    def initialize(self):
        weight_init(self)

####################################### Contrast Texture ###########################################
class Contrast_Block_Deep(nn.Module):
    """ local-context contrasted (LCC) """

    def __init__(self, planes, d1=4, d2=8):
        super(Contrast_Block_Deep, self).__init__()
        self.inplanes = int(planes)
        self.outplanes = int(planes / 2)

        self.local_1 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=1, dilation=1)
        self.context_1 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=d1, dilation=d1)

        self.local_2 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=1, dilation=1)
        self.context_2 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=d2, dilation=d2)

        self.bn1 = nn.BatchNorm2d(self.outplanes)
        self.bn2 = nn.BatchNorm2d(self.outplanes)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.ca = nn.ModuleList([
            CoordAtt(self.outplanes, self.outplanes),
            CoordAtt(self.outplanes, self.outplanes),
            CoordAtt(self.outplanes, self.outplanes),
            CoordAtt(self.outplanes, self.outplanes)
        ])

    def forward(self, x):
        local_1 = self.local_1(x)
        local_1 = self.ca[0](local_1)
        context_1 = self.context_1(x)
        context_1 = self.ca[1](context_1)
        ccl_1 = local_1 - context_1
        ccl_1 = self.bn1(ccl_1)
        ccl_1 = self.relu1(ccl_1)

        local_2 = self.local_2(x)
        local_2 = self.ca[2](local_2)
        context_2 = self.context_2(x)
        context_2 = self.ca[3](context_2)
        ccl_2 = local_2 - context_2
        ccl_2 = self.bn2(ccl_2)
        ccl_2 = self.relu2(ccl_2)

        out = torch.cat((ccl_1, ccl_2), 1)

        return out

    def initialize(self):
        weight_init(self)

########################################### CoordAttention #########################################
# Revised from: Coordinate Attention for Efficient Mobile Network Design, CVPR21
# https://github.com/houqb/CoordAttention
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

    def initialize(self):
        weight_init(self)


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

    def initialize(self):
        weight_init(self)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

    def initialize(self):
        weight_init(self)

# Feature Fusion Module
class FFM(nn.Module):
    def __init__(self, channel):
        super(FFM, self).__init__()
        self.conv_1 = conv3x3(channel, channel)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = conv3x3(channel, channel)
        self.bn_2 = nn.BatchNorm2d(channel)

    def forward(self, x_1, x_2):
        out = x_1 * x_2
        # out = torch.cat((x_1, x_2), dim=1)
        out = F.relu(self.bn_1(self.conv_1(out)), inplace=True)
        out = F.relu(self.bn_2(self.conv_2(out)), inplace=True)
        return out

    def initialize(self):
        weight_init(self)

# Cross Aggregation Module
class CAM(nn.Module):
    def __init__(self, channel):
        super(CAM, self).__init__()
        self.down = nn.Sequential(
            conv3x3(channel, channel, stride=2),
            nn.BatchNorm2d(channel)
        )
        self.conv_1 = conv3x3(channel, channel)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = conv3x3(channel, channel)
        self.bn_2 = nn.BatchNorm2d(channel)
        self.mul = FFM(channel)

    def forward(self, x_high, x_low):
        left_1 = x_low
        left_2 = F.relu(self.down(x_low), inplace=True)
        right_1 = F.interpolate(x_high, size=x_low.size()[2:], mode='bilinear', align_corners=True)
        right_2 = x_high
        left = F.relu(self.bn_1(self.conv_1(left_1 * right_1)), inplace=True)
        right = F.relu(self.bn_2(self.conv_2(left_2 * right_2)), inplace=True)
        right = F.interpolate(right, size=x_low.size()[2:], mode='bilinear', align_corners=True)
        out = self.mul(left, right)
        return out

    def initialize(self):
        weight_init(self)

def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1, bias=False):
    "3x3 convolution with padding"

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=bias)

class OutPut(nn.Module):
    def __init__(self, in_chs, scale=1):
        super(OutPut, self).__init__()
        self.out = nn.Sequential(nn.Conv2d(in_chs, in_chs, 1, bias=False),
                                 nn.BatchNorm2d(in_chs),
                                 nn.ReLU(inplace=True),
                                 nn.UpsamplingBilinear2d(scale_factor=scale),
                                 nn.Conv2d(in_chs, 1, 1),
                                 nn.Sigmoid())

    def forward(self, feat):
        return self.out(feat)

    def initialize(self):
        weight_init(self)

class Decoder(nn.Module):
    def __init__(self, channels):
        super(Decoder, self).__init__()
        self.pyramid_pooling = PyramidPooling(512, 64)
        self.rfb1 = RFB_modified(512, 64)
        self.rfb2 = RFB_modified(320, 64)
        self.conv1 = basicConv(128, 64, k=1, s=1, p=0)
        self.conv2 = basicConv(64, 64, k=1, s=1, p=0)
        self.contrast1 = Contrast_Block_Deep(64)
        self.contrast2 = Contrast_Block_Deep(64)

        self.fusion1 = FFM(64)
        self.fusion2 = FFM(64)
        self.fusion3 = FFM(64)
        self.aggregation1 = CAM(64)
        self.aggregation2 = CAM(64)
        self.img_size = 384
        self.out1 = nn.Sequential(conv3x3(64, 1, bias=True), nn.Sigmoid())
        self.out2 = nn.Sequential(conv3x3(64, 1, bias=True), nn.Sigmoid())
        self.out3 = nn.Sequential(conv3x3(64, 1, bias=True), nn.Sigmoid())
        self.out4 = nn.Sequential(conv3x3(64, 1, bias=True), nn.Sigmoid())
        self.out5 = nn.Sequential(conv3x3(64, 1, bias=True), nn.Sigmoid())
        self.initialize()

    def forward(self, E1, E2, E3, E4, shape):# 64 96, 128 48, 320 24, 512 12
        # same channel
        S1_1 = self.pyramid_pooling(E1)
        S1_2 = self.rfb1(E1)
        S1_3 = self.rfb2(E2)
        S1_4 = self.contrast1(self.conv1(E3))
        S1_5 = self.contrast2(self.conv2(E4))

        S2_1 = self.fusion1(S1_2, F.interpolate(S1_1, size=S1_2.size()[2:], mode='bilinear', align_corners=True))
        S2_2 = self.fusion2(S1_3, F.interpolate(S1_2, size=S1_3.size()[2:], mode='bilinear', align_corners=True))
        S2_3 = self.aggregation1(F.interpolate(S2_1, size=[S1_4.size(2) // 2, S1_4.size(3) // 2], mode='bilinear', align_corners=True), S1_4)
        S2_4 = self.aggregation2(F.interpolate(S2_2, size=[S1_5.size(2) // 2, S1_5.size(3) // 2], mode='bilinear', align_corners=True), S1_5)
        S2_5 = self.fusion3(S2_4, F.interpolate(S2_3, size=S2_4.size()[2:], mode='bilinear', align_corners=True))

        P1 = self.out5(S2_5)
        P2 = self.out4(S2_4)
        P3 = self.out3(S2_3)
        P4 = self.out2(S2_2)
        P5 = self.out1(S2_1)

        P1 = F.interpolate(P1, size=shape, mode='bilinear')
        P2 = F.interpolate(P2, size=shape, mode='bilinear')
        P3 = F.interpolate(P3, size=shape, mode='bilinear')
        P4 = F.interpolate(P4, size=shape, mode='bilinear')
        P5 = F.interpolate(P5, size=shape, mode='bilinear')
        return P5, P4, P3, P2, P1

    def initialize(self):
        weight_init(self)
