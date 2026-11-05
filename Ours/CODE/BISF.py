"""RGB-T 双向交互空间融合。

对两路 backbone 最后一层做双向空间交叉注意力、深度卷积和自注意力，
再拼成一张特征写回 backbone 字典。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


M1_DIM = 256
M1_HEADS = 8


def _map_to_tokens(x: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, HW, C)。"""
    return x.flatten(2).transpose(1, 2)


def _tokens_to_map(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """(B, HW, C) -> (B, C, H, W)。"""
    batch, _, channels = tokens.shape
    return tokens.transpose(1, 2).reshape(batch, channels, height, width)


def _split_heads(tokens: torch.Tensor, num_heads: int) -> torch.Tensor:
    """(B, N, C) -> (B, heads, N, head_dim)。"""
    batch, length, channels = tokens.shape
    head_dim = channels // num_heads
    return tokens.view(batch, length, num_heads, head_dim).transpose(1, 2)


def _merge_heads(tokens: torch.Tensor) -> torch.Tensor:
    """(B, heads, N, head_dim) -> (B, N, C)。"""
    batch, num_heads, length, head_dim = tokens.shape
    return tokens.transpose(1, 2).reshape(batch, length, num_heads * head_dim)


def _attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int) -> torch.Tensor:
    """q、k、v 是 (B, N, C)，返回 (B, Nq, C)。"""
    out = F.scaled_dot_product_attention(
        _split_heads(q, num_heads),
        _split_heads(k, num_heads),
        _split_heads(v, num_heads),
    )
    return _merge_heads(out)


class BidirectionalCrossAttention(nn.Module):
    """双向空间交叉注意力。输出直接替换输入，不加残差。"""

    def __init__(self, dim: int = M1_DIM, num_heads: int = M1_HEADS):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} 必须能被 num_heads {num_heads} 整除")
        self.num_heads = num_heads
        self.norm_rgb = nn.LayerNorm(dim)
        self.norm_thermal = nn.LayerNorm(dim)
        self.q_rgb = nn.Linear(dim, dim)
        self.kv_thermal = nn.Linear(dim, dim * 2)
        self.q_thermal = nn.Linear(dim, dim)
        self.kv_rgb = nn.Linear(dim, dim * 2)
        self.out_rgb = nn.Linear(dim, dim)
        self.out_thermal = nn.Linear(dim, dim)

    def _cross(
        self,
        query_tokens: torch.Tensor,
        memory_tokens: torch.Tensor,
        pos_tokens: torch.Tensor,
        q_proj: nn.Linear,
        kv_proj: nn.Linear,
        out_proj: nn.Linear,
    ) -> torch.Tensor:
        q = q_proj(query_tokens)
        k, v = kv_proj(memory_tokens).chunk(2, dim=-1)
        pos_tokens = pos_tokens.to(dtype=q.dtype)
        q = q + pos_tokens
        k = k + pos_tokens
        return out_proj(_attend(q, k, v, self.num_heads))

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = rgb.shape[-2:]
        rgb_tokens = self.norm_rgb(_map_to_tokens(rgb))
        thermal_tokens = self.norm_thermal(_map_to_tokens(thermal))
        pos_tokens = _map_to_tokens(pos)
        rgb_out = self._cross(rgb_tokens, thermal_tokens, pos_tokens, self.q_rgb, self.kv_thermal, self.out_rgb)
        thermal_out = self._cross(thermal_tokens, rgb_tokens, pos_tokens, self.q_thermal, self.kv_rgb, self.out_thermal)
        return _tokens_to_map(rgb_out, height, width), _tokens_to_map(thermal_out, height, width)


class DepthwiseConv2d(nn.Module):
    """3×3 深度卷积。输出直接替换输入，不加残差。"""

    def __init__(self, dim: int = M1_DIM):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dw(x)


class SpatialSelfAttention(nn.Module):
    """Pre-LN 空间自注意力。输出直接替换输入，不加残差。"""

    def __init__(self, dim: int = M1_DIM, num_heads: int = M1_HEADS):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} 必须能被 num_heads {num_heads} 整除")
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        tokens = self.norm(_map_to_tokens(x))
        q, k, v = self.qkv(tokens).chunk(3, dim=-1)
        pos_tokens = _map_to_tokens(pos).to(dtype=q.dtype)
        q = q + pos_tokens
        k = k + pos_tokens
        return _tokens_to_map(self.out(_attend(q, k, v, self.num_heads)), height, width)


class M1(nn.Module):
    """融合 RGB 与 T 的视觉 backbone 特征。

    参数:
        rgb_backbone_out: RGB LoRA 的 ``backbone.forward_image`` 输出。
        t_backbone_out: T LoRA 的同结构输出。

    两者内部是 Tensor、list、dict 或 None，键包括 ``vision_features``、
    ``vision_pos_enc``、``backbone_fpn`` 和 ``sam2_backbone_out``。
    文本特征不进入本模块。backbone 权重以 bf16 存储，输入张量通常也是 bf16。

    只改 ``backbone_fpn`` 最后一层。scalp 之后这一层是 72×72、256 维，
    位置编码用 RGB 的 ``vision_pos_enc[-1]``，加在 Q 和 K 上。
    双向交叉注意力、深度卷积和空间自注意力都不加残差。
    融合是 ``fused = (R0 + T0) / 2 + Conv1x1(cat(R, T))``。
    ``R0``、``T0`` 是进入交叉注意力之前的最后一层特征，1×1 的权重和偏置初始化为 0。

    返回同结构的新字典，供 ``forward_grounding`` 使用。
    ``vision_features`` 和 ``backbone_fpn[-1]`` 是同一个 fused。
    前几层 FPN、位置编码和 ``sam2_backbone_out`` 沿用 RGB。不修改输入字典。
    返回张量连着本模块已注册的参数，训练才会更新 M1。
    """

    def __init__(self, dim: int = M1_DIM, num_heads: int = M1_HEADS):
        super().__init__()
        self.dim = dim
        self.cross = BidirectionalCrossAttention(dim, num_heads)
        self.dw_rgb = DepthwiseConv2d(dim)
        self.dw_thermal = DepthwiseConv2d(dim)
        self.self_rgb = SpatialSelfAttention(dim, num_heads)
        self.self_thermal = SpatialSelfAttention(dim, num_heads)
        self.fuse = nn.Conv2d(dim * 2, dim, kernel_size=1)
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)

    def forward(self, rgb_backbone_out: dict, t_backbone_out: dict) -> dict:
        rgb_fpn = rgb_backbone_out["backbone_fpn"]
        thermal_fpn = t_backbone_out["backbone_fpn"]
        rgb_pos = rgb_backbone_out["vision_pos_enc"]
        if len(rgb_fpn) == 0 or len(thermal_fpn) == 0 or len(rgb_pos) == 0:
            raise RuntimeError("backbone_fpn 和 vision_pos_enc 不能为空。")

        rgb = rgb_fpn[-1]
        thermal = thermal_fpn[-1]
        pos = rgb_pos[-1]
        if rgb.shape != thermal.shape:
            raise RuntimeError(
                f"RGB 与 T 的最后一层形状应一致，收到 {tuple(rgb.shape)} 和 {tuple(thermal.shape)}"
            )
        if rgb.dim() != 4 or rgb.shape[1] != self.dim:
            raise RuntimeError(f"最后一层应为 (B, {self.dim}, H, W)，收到 {tuple(rgb.shape)}")
        if pos.shape != rgb.shape:
            raise RuntimeError(
                f"位置编码形状应与最后一层一致，收到 {tuple(pos.shape)} 和 {tuple(rgb.shape)}"
            )

        rgb_aligned, thermal_aligned = self.cross(rgb, thermal, pos)
        rgb_refined = self.self_rgb(self.dw_rgb(rgb_aligned), pos)
        thermal_refined = self.self_thermal(self.dw_thermal(thermal_aligned), pos)
        fused = (rgb + thermal) / 2 + self.fuse(torch.cat([rgb_refined, thermal_refined], dim=1))

        output = dict(rgb_backbone_out)
        fpn = list(rgb_fpn)
        fpn[-1] = fused
        output["backbone_fpn"] = fpn
        output["vision_features"] = fused
        return output
