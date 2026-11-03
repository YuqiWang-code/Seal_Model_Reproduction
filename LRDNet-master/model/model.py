import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone.get_backbone import get_backbone

def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU) or isinstance(m, nn.GELU) or isinstance(m, nn.LeakyReLU) or isinstance(m,
                                                                                                           nn.AdaptiveAvgPool2d) or isinstance(
            m, nn.ReLU6) or isinstance(m, nn.MaxPool2d) or isinstance(m, nn.Softmax):
            pass
        elif isinstance(m, nn.ModuleList):
            weight_init(m)
        else:
            m.initialize()


class DyReLU(nn.Module):
    def __init__(self, num_func=2, scale=2., serelu=False):
        """
        num_func: -1: none
                   0: relu
                   1: SE
                   2: dy-relu
        """
        super(DyReLU, self).__init__()

        assert (-1 <= num_func <= 2)
        self.num_func = num_func
        self.scale = scale

        serelu = serelu and num_func == 1
        self.act = nn.ReLU6(inplace=True) if num_func == 0 or serelu else nn.Sequential()

    def forward(self, x):
        if isinstance(x, tuple):
            out, a = x
        else:
            out = x

        out = self.act(out)

        if self.num_func == 1:  # SE
            a = a * self.scale
            out = out * a
        elif self.num_func == 2:  # DY-ReLU
            _, C, _, _ = a.shape
            a1, a2 = torch.split(a, [C // 2, C // 2], dim=1)
            a1 = (a1 - 0.5) * self.scale + 1.0  # 0.0 -- 2.0
            a2 = (a2 - 0.5) * self.scale  # -1.0 -- 1.0
            out = torch.max(out * a1, out * a2)

        return out

    def initialize(self):
        weight_init(self)


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=bias)


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


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias)


class BRM(nn.Module):
    def __init__(self, channel):
        super(BRM, self).__init__()
        self.conv_atten = conv1x1(channel, channel)
        self.conv_1 = conv3x3(channel, channel)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = conv3x3(channel, channel)
        self.bn_2 = nn.BatchNorm2d(channel)

    def forward(self, x_1, x_edge):
        # x = torch.cat([x_1, x_edge], dim=1)
        x = x_1 + x_edge
        atten = F.avg_pool2d(x, x.size()[2:])
        atten = torch.sigmoid(self.conv_atten(atten))
        out = torch.mul(x, atten) + x
        out = F.relu(self.bn_1(self.conv_1(out)), inplace=True)
        out = F.relu(self.bn_2(self.conv_2(out)), inplace=True)
        return out

    def initialize(self):
        weight_init(self)


# Revised from: PraNet: Parallel Reverse Attention Network for Polyp Segmentation, MICCAI20
# https://github.com/DengPingFan/PraNet
class RFB_modified(nn.Module):
    """ logical semantic relation (LSR) """

    def __init__(self, in_channel):
        super(RFB_modified, self).__init__()
        out_channel = in_channel

        self.relu = DyReLU()
        self.branch0 = nn.Sequential(
            basicConv(in_channel, out_channel, k=1, p=0, relu=False),
        )
        self.branch1 = nn.Sequential(
            basicConv(in_channel, out_channel, k=1, p=0),
            basicConv(out_channel, out_channel, k=7, p=3),
            basicConv(out_channel, out_channel, 3, p=7, d=7, relu=False)
        )
        self.branch2 = nn.Sequential(
            basicConv(in_channel, out_channel, k=1, p=0),
            basicConv(out_channel, out_channel, k=7, p=3),
            basicConv(out_channel, out_channel, k=7, p=3),
            basicConv(out_channel, out_channel, 3, p=7, d=7, relu=False)
        )
        self.branch3 = nn.Sequential(
            basicConv(in_channel, out_channel, k=1, p=0),
            basicConv(out_channel, out_channel, k=7, p=3),
            basicConv(out_channel, out_channel, k=7, p=3),
            basicConv(out_channel, out_channel, 3, p=7, d=7, relu=False)
        )
        self.conv_cat = basicConv(4 * out_channel, out_channel, 3, p=1, relu=False)
        self.conv_res = basicConv(in_channel, out_channel, 1, p=0, relu=False)

    def forward(self, x, a):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu((x_cat + self.conv_res(x), a))
        return x

    def initialize(self):
        weight_init(self)


############################################# Pooling ##############################################
class basicConv(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True, dy_relu=False):
        super(basicConv, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
            # conv.append(nn.LayerNorm(out_channel, eps=1e-6))
        if relu:
            conv.append(nn.GELU())
        self.conv = nn.Sequential(*conv)
        self.dy_relu = dy_relu
        self.act = DyReLU()

    def forward(self, x):
        if self.dy_relu:
            feature, a = x
            out = self.conv(feature)
            return self.act((out, a))
        else:
            return self.conv(x)

    def initialize(self):
        weight_init(self)


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

        self.relu = DyReLU()

        self.ca = nn.ModuleList([
            CoordAtt(self.outplanes, self.outplanes),
            CoordAtt(self.outplanes, self.outplanes),
            CoordAtt(self.outplanes, self.outplanes),
            CoordAtt(self.outplanes, self.outplanes)
        ])

    def forward(self, x, a):
        local_1 = self.local_1(x)
        local_1 = self.ca[0](local_1)
        context_1 = self.context_1(x)
        context_1 = self.ca[1](context_1)
        ccl_1 = local_1 - context_1
        ccl_1 = self.bn1(ccl_1)

        local_2 = self.local_2(x)
        local_2 = self.ca[2](local_2)
        context_2 = self.context_2(x)
        context_2 = self.ca[3](context_2)
        ccl_2 = local_2 - context_2
        ccl_2 = self.bn2(ccl_2)

        out = torch.cat((ccl_1, ccl_2), 1)
        out = self.relu((out, a))

        return out

    def initialize(self):
        weight_init(self)


####################################### ConvTrans ###########################################

class HyperFunc1(nn.Module):
    def __init__(self, token_dim, oup, reduction_ratio=4):
        super(HyperFunc1, self).__init__()

        squeeze_dim = token_dim // reduction_ratio
        self.hyper = nn.Sequential(
            nn.Linear(token_dim, squeeze_dim),
            nn.ReLU(inplace=True),
            nn.Linear(squeeze_dim, oup),
            h_sigmoid()
        )

    def forward(self, x):
        t = x.mean(dim=0)
        h = self.hyper(t)
        h = torch.unsqueeze(torch.unsqueeze(h, 2), 3)
        return h

    def initialize(self):
        weight_init(self)


class HyperFunc2(nn.Module):
    def __init__(self, token_dim, oup, reduction_ratio=4):
        super(HyperFunc2, self).__init__()

        squeeze_dim = token_dim // reduction_ratio
        self.hyper = nn.Sequential(
            nn.Linear(token_dim, squeeze_dim),
            nn.ReLU(inplace=True),
            nn.Linear(squeeze_dim, oup),
        )

    def forward(self, x, attn):
        hp = self.hyper(x).permute(1, 2, 0)  # bs x hyper_dim x T
        attn = attn.mean(dim=1)
        bs, T, H, W = attn.shape
        attn = attn.view(bs, T, H * W)
        hp = torch.matmul(hp, attn).softmax(dim=-1)  # bs x hyper_dim x HW
        h = hp.view(bs, -1, H, W)
        return h

    def initialize(self):
        weight_init(self)


class Local2Global(nn.Module):
    def __init__(self, inp, token_dim=128, token_num=6, attn_num_heads=2):
        super(Local2Global, self).__init__()

        self.num_heads = attn_num_heads
        self.token_num = token_num

        self.scale = (inp // attn_num_heads) ** -0.5
        self.q = nn.Linear(token_dim, inp)

        self.proj = nn.Linear(inp, token_dim)
        self.layer_norm = nn.LayerNorm(token_dim)

    def forward(self, x):
        features, tokens = x  # features: bs x C x H x W tokens: T x bs x Ct
        bs, C, H, W = features.shape
        T, _, _ = tokens.shape

        t = self.q(tokens).view(T, bs, self.num_heads, -1).permute(1, 2, 0, 3)  # from T x bs x Ct to bs x N x T x Ct/N
        k = features.view(bs, self.num_heads, -1, H * W)  # bs x N x C/N x HW
        attn = (t @ k) * self.scale  # bs x N x T x HW
        attn_out = attn.softmax(dim=-1)  # bs x N x T x HW
        attn_out = (attn_out @ k.transpose(-1, -2))  # bs x N x T x C/N (k: bs x N x C/N x HW)
        # note here: k=v without transform
        t_a = attn_out.permute(2, 0, 1, 3)  # T x bs x N x C/N
        t_a = t_a.reshape(T, bs, -1)
        t_a = self.proj(t_a)
        tokens = tokens + t_a
        tokens = self.layer_norm(tokens)
        bs, Nh, Ca, HW = attn.shape
        attn = attn.view(bs, Nh, Ca, H, W)
        return tokens, attn

    def initialize(self):
        weight_init(self)


class GlobalBlock(nn.Module):
    def __init__(
            self, token_dim=128, token_num=6, attn_num_heads=4):
        super(GlobalBlock, self).__init__()

        self.num_heads = attn_num_heads
        self.token_num = token_num
        self.ffn_exp = 2

        self.ffn = nn.Sequential(
            nn.Linear(token_dim, token_dim * self.ffn_exp),
            nn.GELU(),
            nn.Linear(token_dim * self.ffn_exp, token_dim)
        )
        self.ffn_norm = nn.LayerNorm(token_dim)

        self.scale = (token_dim // attn_num_heads) ** -0.5
        self.q = nn.Linear(token_dim, token_dim)

        self.channel_mlp = nn.Linear(token_dim, token_dim)
        self.layer_norm = nn.LayerNorm(token_dim)

    def forward(self, x):
        tokens = x
        T, bs, C = tokens.shape
        t = self.q(tokens).view(T, bs, self.num_heads, -1).permute(1, 2, 0, 3)  # from T x bs x Ct to bs x N x T x Ct/N
        k = tokens.permute(1, 2, 0).view(bs, self.num_heads, -1,
                                         T)  # from T x bs x Ct -> bs x Ct x T -> bs x N x Ct/N x T
        attn = (t @ k) * self.scale  # bs x N x T x T

        attn_out = attn.softmax(dim=-1)  # bs x N x T x T
        attn_out = (attn_out @ k.transpose(-1, -2))  # bs x N x T x C/N (k: bs x N x Ct/N x T)
        # note here: k=v without transform
        t_a = attn_out.permute(2, 0, 1, 3)  # T x bs x N x C/N
        t_a = t_a.reshape(T, bs, -1)

        t_a = self.channel_mlp(t_a)  # token_num x bs x channel
        tokens = tokens + t_a
        tokens = self.layer_norm(tokens)

        t_ffn = self.ffn(tokens)
        tokens = tokens + t_ffn
        tokens = self.ffn_norm(tokens)
        return tokens

    def initialize(self):
        weight_init(self)


class Global2Local(nn.Module):
    def __init__(self, inp, token_dim=128, token_num=6, attn_num_heads=2):
        super(Global2Local, self).__init__()

        self.token_num = token_num
        self.num_heads = attn_num_heads
        self.scale = (inp // attn_num_heads) ** -0.5
        self.k = nn.Linear(token_dim, inp)
        self.proj = nn.Linear(token_dim, inp)

    def forward(self, x):
        out, tokens = x
        v = self.proj(tokens).permute(1, 2, 0)  # from T x bs x Ct -> T x bs x C -> bs x C x T
        bs, C, H, W = out.shape
        q = out.view(bs, self.num_heads, -1, H * W).transpose(-1, -2)
        k = self.k(tokens).permute(1, 2, 0).view(bs, self.num_heads, -1,
                                                 self.token_num)  # from T x bs x Ct -> bs x C x T -> bs x N x C/N x T
        attn = (q @ k) * self.scale  # bs x N x HW x T
        attn_out = attn.softmax(dim=-1)  # bs x N x HW x T

        vh = v.view(bs, self.num_heads, -1, self.token_num)  # bs x N x C/N x T
        attn_out = (attn_out @ vh.transpose(-1, -2))  # bs x N x HW x C/N
        # note here k != v
        g_a = attn_out.transpose(-1, -2).reshape(bs, C, H, W)  # bs x C x HW
        out = out + g_a
        return out

    def initialize(self):
        weight_init(self)


class TransDecorators(nn.Module):
    def __init__(self, planes, ConvBlock, token_dim=128, token_num=3, attn_num_heads=2):  # RFB/LCC
        super(TransDecorators, self).__init__()

        self.local_global = Local2Global(planes, token_dim=token_dim, token_num=token_num,
                                         attn_num_heads=attn_num_heads)
        self.global_block = GlobalBlock(token_dim=token_dim, token_num=token_num)

        self.global_local = Global2Local(planes, token_num=token_num, attn_num_heads=attn_num_heads)
        self.conv_block = ConvBlock(planes)
        self.rec_attn = HyperFunc2(token_dim, planes)
        self.hyper = HyperFunc1(token_dim, token_dim)

    def forward(self, features, tokens):
        tokens, attn = self.local_global((features, tokens))
        tokens = self.global_block(tokens)
        hp = self.hyper(tokens)
        features = self.conv_block(features, hp)
        out = self.global_local((features, tokens))
        rec_attn = self.rec_attn(tokens, attn)
        return out + features, tokens, rec_attn,


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        b, c, h, w = x.size()

        channels_per_group = c // self.groups

        # reshape
        x = x.view(b, self.groups, channels_per_group, h, w)

        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        out = x.view(b, -1, h, w)
        return out

    def initialize(self):
        pass


class RUpsampling(nn.Module):
    def __init__(self, inp):
        super(RUpsampling, self).__init__()
        self.conv1 = basicConv(inp, inp, k=1, p=0)
        self.conv2 = basicConv(2 * inp, inp, k=1, p=0, g=inp)
        self.shuffle = ChannelShuffle(inp)

    def forward(self, x, rec_attn):
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        res = x
        x = x * rec_attn
        out = torch.concat((x, res), dim=1)
        out = self.conv2(out)
        return self.shuffle(out)

    def initialize(self):
        weight_init(self)


################################################ Net ###############################################
class Net(nn.Module):
    def __init__(self, option, token_num=3, token_dim=128):
        super(Net, self).__init__()

        self.tokens = nn.Embedding(token_num, token_dim)

        self.bkbone, _ = get_backbone(option)
        self.pyramid_pooling = PyramidPooling(2048, 64)

        self.conv1 = nn.ModuleList([
            basicConv(64, 64, k=1, s=1, p=0),
            basicConv(256, 64, k=1, s=1, p=0),
            basicConv(512, 64, k=1, s=1, p=0),
            basicConv(1024, 64, k=1, s=1, p=0),
            basicConv(2048, 64, k=1, s=1, p=0),
        ])

        self.rfb = nn.ModuleList([
            TransDecorators(64, RFB_modified),
            TransDecorators(64, RFB_modified)
        ])

        self.contrast = nn.ModuleList([
            TransDecorators(64, Contrast_Block_Deep),
            TransDecorators(64, Contrast_Block_Deep)
        ])

        self.fusion = nn.ModuleList([
            FFM(64),
            FFM(64),
            FFM(64),
            FFM(64)
        ])

        self.aggregation = nn.ModuleList([
            CAM(64),
            CAM(64)
        ])

        self.refine = BRM(64)
        # self.conv2 = basicConv(128, 64, k=1, s=1, p=0)

        self.edge_head = nn.Conv2d(64, 1, 3, 1, 1)
        self.head = nn.ModuleList([
            conv3x3(64, 1, bias=True),
            conv3x3(64, 1, bias=True),
            conv3x3(64, 1, bias=True),
            conv3x3(64, 1, bias=True),
            conv3x3(64, 1, bias=True),
        ])

        self.rup = nn.ModuleList([
            RUpsampling(64),
            RUpsampling(64),
            RUpsampling(64),
            RUpsampling(64),
        ])

        self.initialize()

    def forward(self, x, shape=None, name=None):
        bs, C, H, W = x.shape
        if shape is not None:
            H, W = shape
        tokens = self.tokens.weight
        x0_t = tokens[None].repeat(bs, 1, 1).clone().permute(1, 0, 2)

        bk_stage2, bk_stage3, bk_stage4, bk_stage5 = self.bkbone(x)
        f_c3 = self.pyramid_pooling(bk_stage5)
        bk_stage2, bk_stage3, bk_stage4, bk_stage5 = self.conv1[1](bk_stage2), self.conv1[2](bk_stage3), self.conv1[3](
            bk_stage4), self.conv1[4](bk_stage5)

        f_c2, x1_t, rec_attn1 = self.rfb[0](bk_stage5, x0_t)
        f_c1, x2_t, rec_attn2 = self.rfb[1](bk_stage4, x1_t)
        f_t2, x3_t, rec_attn3 = self.contrast[0](bk_stage3, x2_t)
        f_t1, x4_t, rec_attn4 = self.contrast[1](bk_stage2, x3_t)

        fused3 = self.fusion[2](f_c2, f_c3)

        fused2 = self.rup[0](f_c2, rec_attn2)
        fused2 = self.fusion[1](f_c1, fused2)

        a2 = self.rup[1](fused3, rec_attn2)
        a2 = self.aggregation[1](a2, f_t2)

        a1 = self.rup[2](fused2, rec_attn3)
        a1 = self.aggregation[0](a1, f_t1)

        a2 = self.rup[2](a2, rec_attn4)
        out0 = self.fusion[0](a1, a2)

        out0 = F.interpolate(self.head[0](out0), size=(H, W), mode='bilinear', align_corners=False)

        if self.training:
            out1 = F.interpolate(self.head[1](a1), size=(H, W), mode='bilinear', align_corners=False)
            out2 = F.interpolate(self.head[2](a2), size=(H, W), mode='bilinear', align_corners=False)
            out3 = F.interpolate(self.head[3](fused2), size=(H, W), mode='bilinear', align_corners=False)
            out4 = F.interpolate(self.head[4](fused3), size=(H, W), mode='bilinear', align_corners=False)
            return out0, None, out1, out2, out3, out4
        else:
            return out0, None

    def initialize(self):
        pass


def get_model(option):
    model = Net(option=option).cuda()

    if option['checkpoint'] is not None:
        model.load_state_dict(torch.load(option['checkpoint']))
        print('Load checkpoint from {}'.format(option['checkpoint']))
    return model

# def gen_feature_heat(x_show, position, name, H, W):
#     x_show = F.interpolate(x_show, size=(H, W), mode='bilinear', align_corners=False)
#     x_show = torch.mean(x_show, dim=1, keepdim=True).sigmoid().data.cpu().numpy().squeeze()
#     x_show = (x_show - x_show.min()) / (x_show.max() - x_show.min() + 1e-8)*255
#     os.makedirs('./heat/' + position, exist_ok=True)
#
#     cv2.imwrite('./heat/' + position + '/'+name, x_show)


