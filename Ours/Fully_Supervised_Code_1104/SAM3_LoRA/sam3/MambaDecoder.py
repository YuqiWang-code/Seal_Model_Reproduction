"""
MambaDecoder for Noprompt branch.
All dependencies self-contained; no imports from vmamba.py,
SalMamba_dual.py, or the original MambaDecoder.py.

Input:  shallow(B,32,288,288), mid(B,64,144,144), deep(B,256,72,72)
Output: (B,1,288,288) mask logits
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from functools import partial
from einops import rearrange, repeat
from timm.layers import DropPath

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None


# ---------------------------------------------------------------------------
# CrossScan / CrossMerge (4-direction, from vmamba)
# ---------------------------------------------------------------------------

class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs

    @staticmethod
    def backward(ctx, ys):
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)


class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y

    @staticmethod
    def backward(ctx, x):
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs.view(B, 4, C, H, W), None, None


def cross_selective_scan(x, x_proj_weight, x_proj_bias,
                         dt_projs_weight, dt_projs_bias,
                         A_logs, Ds, out_norm, delta_softplus=True):
    B, D, H, W = x.shape
    _, N = A_logs.shape
    K, _, R = dt_projs_weight.shape
    L = H * W
    xs = CrossScan.apply(x)
    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    xs  = xs.view(B, -1, L).float()
    dts = dts.contiguous().view(B, -1, L).float()
    As  = -torch.exp(A_logs.float())
    Bs  = Bs.contiguous().float()
    Cs  = Cs.contiguous().float()
    Ds_ = Ds.float()
    db  = dt_projs_bias.view(-1).float()
    ys = selective_scan_fn(
        xs, dts, As, Bs, Cs, Ds_,
        delta_bias=db, delta_softplus=delta_softplus,
    ).view(B, K, -1, H, W)
    y = CrossMerge.apply(ys)
    y = y.transpose(1, 2).contiguous().view(B, H, W, -1)
    return out_norm(y).to(x.dtype)


# ---------------------------------------------------------------------------
# SS2D (2-D Selective Scan, from vmamba)
# ---------------------------------------------------------------------------

class SS2D(nn.Module):
    def __init__(self, d_model=96, d_state=16, ssm_ratio=2, dt_rank="auto",
                 d_conv=3, conv_bias=True, dropout=0., bias=False,
                 dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0,
                 dt_init_floor=1e-4, softmax_version=False, **kwargs):
        super().__init__()
        self.d_inner = int(ssm_ratio * d_model)
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_conv  = d_conv
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        if d_conv > 1:
            self.conv2d = nn.Conv2d(
                self.d_inner, self.d_inner, groups=self.d_inner,
                bias=conv_bias, kernel_size=d_conv, padding=(d_conv - 1) // 2,
            )
            self.act = nn.SiLU()
        K = 4
        xps = [nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False) for _ in range(K)]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in xps], dim=0))
        dts = [self._dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor) for _ in range(K)]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dts], dim=0))
        self.dt_projs_bias   = nn.Parameter(torch.stack([t.bias   for t in dts], dim=0))
        self.A_logs   = self._A_log_init(self.d_state, self.d_inner, copies=K, merge=True)
        self.Ds       = self._D_init(self.d_inner, copies=K, merge=True)
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        self.dropout  = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    @staticmethod
    def _dt_init(dt_rank, d_inner, dt_scale=1., dt_init="random",
                 dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        p = nn.Linear(dt_rank, d_inner, bias=True)
        std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(p.weight, std)
        else:
            nn.init.uniform_(p.weight, -std, std)
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        with torch.no_grad():
            p.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        return p

    @staticmethod
    def _A_log_init(d_state, d_inner, copies=1, merge=True):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32),
                   "n -> d n", d=d_inner).contiguous()
        A_log = torch.log(A)
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def _D_init(d_inner, copies=1, merge=True):
        D = torch.ones(d_inner)
        if copies > 0:
            D = repeat(D, "n -> r n", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward(self, x):
        xz = self.in_proj(x)
        if self.d_conv > 1:
            x_, z = xz.chunk(2, dim=-1)
            x_ = self.act(self.conv2d(x_.permute(0, 3, 1, 2).contiguous()))
            y  = cross_selective_scan(x_, self.x_proj_weight, None,
                                      self.dt_projs_weight, self.dt_projs_bias,
                                      self.A_logs, self.Ds, self.out_norm)
            y  = y * F.silu(z)
        else:
            xz = F.silu(xz)
            x_, z = xz.chunk(2, dim=-1)
            x_ = x_.permute(0, 3, 1, 2).contiguous()
            y  = cross_selective_scan(x_, self.x_proj_weight, None,
                                      self.dt_projs_weight, self.dt_projs_bias,
                                      self.A_logs, self.Ds, self.out_norm)
            y  = y * z
        return self.dropout(self.out_proj(y))


# ---------------------------------------------------------------------------
# ChannelAttention / ChannelAttentionBlock / CVSSDecoderBlock (from vmamba)
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class ChannelAttentionBlock(nn.Module):
    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super().__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor),
        )

    def forward(self, x):
        return self.cab(x)


class CVSSDecoderBlock(nn.Module):
    def __init__(self, hidden_dim=0, drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 attn_drop_rate=0., d_state=16, dt_rank="auto",
                 ssm_ratio=2.0, shared_ssm=False, softmax_version=False,
                 use_checkpoint=False, mlp_ratio=0., act_layer=nn.GELU,
                 drop=0., **kwargs):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1  = norm_layer(hidden_dim)
        self.scale1 = nn.Parameter(torch.ones(hidden_dim))
        self.op     = SS2D(d_model=hidden_dim, dropout=attn_drop_rate,
                           d_state=d_state, ssm_ratio=ssm_ratio,
                           dt_rank=dt_rank, softmax_version=softmax_version,
                           **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.conv_blk  = ChannelAttentionBlock(hidden_dim)
        self.norm2  = norm_layer(hidden_dim)
        self.scale2 = nn.Parameter(torch.ones(hidden_dim))

    def _forward(self, x):
        x = x * self.scale1 + self.drop_path(self.op(self.norm1(x)))
        y = self.conv_blk(
            self.norm2(x).permute(0, 3, 1, 2).contiguous()
        ) + (x * self.scale2).permute(0, 3, 1, 2).contiguous()
        return y.permute(0, 2, 3, 1).contiguous()

    def forward(self, x):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x)
        return self._forward(x)


# ---------------------------------------------------------------------------
# Helpers: ChannelProj, MambaUpBlock
# ---------------------------------------------------------------------------

class ChannelProj(nn.Module):
    """1x1 Conv + LayerNorm, output (B, H, W, out_ch)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1).contiguous()
        return self.norm(x)


class MambaUpBlock(nn.Module):
    """CVSSDecoderBlocks + skip connection + 2x bilinear upsample."""
    def __init__(self, dim, out_dim, depth, drop_path_rate=0.1,
                 d_state=4, use_checkpoint=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            CVSSDecoderBlock(
                hidden_dim=dim, drop_path=drop_path_rate,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                attn_drop_rate=0., d_state=d_state, dt_rank="auto",
                ssm_ratio=2.0, softmax_version=False,
                use_checkpoint=use_checkpoint, mlp_ratio=0.,
                act_layer=nn.GELU, drop=0.,
            )
            for _ in range(depth)
        ])
        self.upsample_proj = nn.Linear(dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, skip):
        """x, skip: (B, H, W, dim) -> (B, 2H, 2W, out_dim)"""
        x = x + skip
        for blk in self.blocks:
            x = blk(x)
        B, H, W, C = x.shape
        x = F.interpolate(
            x.permute(0, 3, 1, 2).contiguous(),
            scale_factor=2, mode='bilinear', align_corners=False,
        ).permute(0, 2, 3, 1).contiguous()
        return self.norm(self.upsample_proj(x))


# ---------------------------------------------------------------------------
# SAMMambaDecoder
# ---------------------------------------------------------------------------

class SAMMambaDecoder(nn.Module):
    """
    3-scale Mamba Decoder for noprompt branch.

    Stage1: deep(256->128)   @ 72x72   -> MambaUpBlock -> 144x144, 64ch
    Stage2: +mid(64)         @ 144x144 -> MambaUpBlock -> 288x288, 32ch
    Stage3: +shallow(32)     @ 288x288 -> CVSSBlocks   -> head -> (B,1,288,288)
    """
    SAM_CH  = [32,  64, 256]   # [shallow, mid, deep]
    HIDE_CH = [32,  64, 128]   # [D0,      D1,  D2  ]

    def __init__(self, depth=2, drop_path_rate=0.1, d_state=4, use_checkpoint=False):
        super().__init__()
        C0, C1, C2 = self.SAM_CH
        D0, D1, D2 = self.HIDE_CH

        self.proj_deep    = ChannelProj(C2, D2)
        self.proj_mid     = ChannelProj(C1, D1)
        self.proj_shallow = ChannelProj(C0, D0)

        self.stage1 = MambaUpBlock(D2, D1, depth, drop_path_rate, d_state, use_checkpoint)
        self.stage2 = MambaUpBlock(D1, D0, depth, drop_path_rate, d_state, use_checkpoint)

        self.stage3 = nn.ModuleList([
            CVSSDecoderBlock(
                hidden_dim=D0, drop_path=drop_path_rate,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                attn_drop_rate=0., d_state=d_state, dt_rank="auto",
                ssm_ratio=2.0, softmax_version=False,
                use_checkpoint=use_checkpoint, mlp_ratio=0.,
                act_layer=nn.GELU, drop=0.,
            )
            for _ in range(depth)
        ])
        self.norm_out = nn.LayerNorm(D0)
        self.head     = nn.Linear(D0, 1, bias=True)

        # HQ head (EG-SAM style)
        # compress_vit_feat: shallow(B,32,288,288) -> (B, D0, 288, 288)
        self.compress_shallow = nn.Sequential(
            nn.Conv2d(C0, D0, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, D0),
            nn.GELU(),
        )
        # embedding_encoder: deep(B,256,72,72) -> upsample -> (B, D0, 288, 288)
        self.compress_deep = nn.Sequential(
            nn.ConvTranspose2d(C2, D2, kernel_size=2, stride=2),   # 72->144
            nn.GroupNorm(8, D2),
            nn.GELU(),
            nn.ConvTranspose2d(D2, D1, kernel_size=2, stride=2),   # 144->288
            nn.GroupNorm(4, D1),
            nn.GELU(),
            nn.Conv2d(D1, D0, kernel_size=1, bias=False),
        )
        # embedding_maskfeature: refine stage3 feat (B,D0,288,288)
        self.hq_maskfeature = nn.Sequential(
            nn.Conv2d(D0, D0 * 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, D0 * 2),
            nn.GELU(),
            nn.Conv2d(D0 * 2, D0, kernel_size=3, padding=1, bias=False),
        )
        # HQ prediction head (residual on top of SAM mask)
        self.hq_head = nn.Conv2d(D0, 1, kernel_size=1, bias=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, shallow, mid, deep):
        """
        shallow: (B,  32, 288, 288)
        mid:     (B,  64, 144, 144)
        deep:    (B, 256,  72,  72)
        returns: (B,   1, 288, 288) mask logits
        """
        f_deep    = self.proj_deep(deep)       # (B,  72,  72, 128)
        f_mid     = self.proj_mid(mid)         # (B, 144, 144,  64)
        f_shallow = self.proj_shallow(shallow) # (B, 288, 288,  32)

        # Stage 1: bottleneck (zero skip at deepest level)
        x = self.stage1(f_deep, torch.zeros_like(f_deep))  # (B,144,144, 64)

        # Stage 2: fuse mid-scale
        x = self.stage2(x, f_mid)              # (B,288,288, 32)

        # Stage 3: fuse shallow, refine
        x = x + f_shallow
        for blk in self.stage3:
            x = blk(x)                         # (B,288,288, 32)
        x = self.norm_out(x)                   # (B,288,288, 32)

        # SAM mask (base)
        x_spatial = x.permute(0, 3, 1, 2).contiguous()  # (B, 32, 288, 288)
        sam_mask = self.head(x)                           # (B,288,288,  1)
        sam_mask = sam_mask.permute(0, 3, 1, 2).contiguous()  # (B,1,288,288)

        # HQ residual (EG-SAM style)
        # hq_features = compress_shallow(shallow) + compress_deep(deep)
        hq_feat = self.compress_shallow(shallow) + self.compress_deep(deep)  # (B,32,288,288)
        # refine stage3 spatial feature and fuse with hq_feat
        hq_upscaled = self.hq_maskfeature(x_spatial) + hq_feat  # (B,32,288,288)
        hq_mask = self.hq_head(hq_upscaled)                      # (B,1,288,288)

        # Final: SAM base + HQ residual
        final_mask = sam_mask + hq_mask                           # (B,1,288,288)
        return final_mask, hq_mask, x_spatial                    # x_spatial for prototype-based refinement
