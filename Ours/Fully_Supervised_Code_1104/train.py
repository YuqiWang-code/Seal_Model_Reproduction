"""全监督 RGB-T 训练。

冻结 SAM3 和两套视觉 LoRA。共享 LoRA3 加在两套模型的视觉 qkv/proj 上，默认与 M1 一起训练。
M1 融合 RGB / T 的图像特征后，配上数据集里的真实框进一次 Decoder。
预测 mask 用数据集 GT 做结构损失、Sigmoid Focal Loss 和 Dice Loss。
传入 --no-lora3 时只更新 M1。
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("SEAL_CUDA", "1"))

import argparse
import importlib.resources
import logging
import sys
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pkg_resources

if not hasattr(pkg_resources, "resource_filename"):

    def _resource_filename(package: str, resource: str) -> str:
        try:
            return str(importlib.resources.files(package).joinpath(resource))
        except Exception:
            return str(importlib.resources.files(package.replace("-", "_")).joinpath(resource))

    pkg_resources.resource_filename = _resource_filename

ROOT = Path(__file__).resolve().parent
LORA_ROOT = ROOT / "SAM3_LoRA"
SAM3_PKG_ROOT = LORA_ROOT / "sam3"
sys.path.insert(0, str(LORA_ROOT))
sys.path.insert(0, str(SAM3_PKG_ROOT))

import torch
torch.multiprocessing.set_sharing_strategy("file_system")
import torch.nn as nn
import torch.nn.functional as F

from sam3.model.data_misc import FindStage
from sam3.model.geometry_encoders import Prompt
from sam3.model_builder import build_sam3_image_model
from lora_layers import LoRAConfig, LoRALayer, LoRALinear, apply_lora_to_model, load_lora_weights

from data import get_loader
from utils import (
    adjust_lr,
    calc_mae,
    clone_tree,
    setup_logger,
    structure_loss,
)


M1_DIM = 256
M1_HEADS = 8
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
LORA3_RANK = 8
LORA3_ALPHA = 16.0
LORA3_DROPOUT = 0.05
LORA3_LAYER_COUNT = 64


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


def _zero_linear(layer: nn.Linear) -> None:
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class BidirectionalCrossAttention(nn.Module):
    """双向空间交叉注意力。输出投影为 0，残差加回原特征。"""

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
        _zero_linear(self.out_rgb)
        _zero_linear(self.out_thermal)

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
        rgb_update = self._cross(rgb_tokens, thermal_tokens, pos_tokens, self.q_rgb, self.kv_thermal, self.out_rgb)
        thermal_update = self._cross(thermal_tokens, rgb_tokens, pos_tokens, self.q_thermal, self.kv_rgb, self.out_thermal)
        return rgb + _tokens_to_map(rgb_update, height, width), thermal + _tokens_to_map(thermal_update, height, width)


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
    双向交叉注意力保留残差，输出投影初始化为 0，第 0 步这一层是恒等映射。
    随后 RGB、T 各自做 3×3 深度卷积和空间自注意力，这两步不加残差。
    融合是 ``fused = Conv1x1(cat(R, T))``，不加 RGB。
    深度卷积、自注意力和 1×1 使用默认初始化，避免没有残差时输出变成全零。

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
        fused = self.fuse(torch.cat([rgb_refined, thermal_refined], dim=1))

        output = dict(rgb_backbone_out)
        fpn = list(rgb_fpn)
        fpn[-1] = fused
        output["backbone_fpn"] = fpn
        output["vision_features"] = fused
        return output


def configure_low_memory_model(model: nn.Module) -> None:
    """关闭 DAC，与 demo8 训练时的 Decoder 设置一致。query 数已是 200。"""
    decoder = model.transformer.decoder
    decoder.dac = False
    decoder.num_o2m_queries = 0
    model.dac = False


def build_frozen_lora_model(checkpoint_path: Path, lora_path: Path, device: torch.device) -> nn.Module:
    """按 demo10 加载一套视觉 LoRA，然后连同 LoRA 一起冻结。"""
    bpe_path = SAM3_PKG_ROOT / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    if not bpe_path.is_file():
        raise FileNotFoundError(f"找不到 SAM3 词表: {bpe_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"找不到 SAM3 权重: {checkpoint_path}")
    if not lora_path.is_file():
        raise FileNotFoundError(f"找不到 LoRA 权重: {lora_path}")

    model = build_sam3_image_model(
        bpe_path=str(bpe_path),
        device="cpu",
        eval_mode=False,
        checkpoint_path=str(checkpoint_path),
        load_from_HF=False,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        compile=False,
    )
    configure_low_memory_model(model)
    lora_cfg = LoRAConfig(
        rank=16,
        alpha=32.0,
        dropout=0.0,
        target_modules=["qkv", "proj"],
        apply_to_vision_encoder=True,
        apply_to_text_encoder=False,
        apply_to_geometry_encoder=False,
        apply_to_detr_encoder=False,
        apply_to_detr_decoder=False,
        apply_to_mask_decoder=False,
    )
    model = apply_lora_to_model(model, lora_cfg)
    load_lora_weights(model, str(lora_path))

    for name, param in model.backbone.named_parameters():
        if "lora" not in name:
            param.data = param.data.to(torch.bfloat16)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    model.to(device)
    return model


class SharedLoRA(nn.Module):
    """两套视觉主干共用的 LoRA3。按模块全名索引到各自的 LoRALayer。"""

    def __init__(self, named_shapes: dict[str, tuple[int, int]]):
        super().__init__()
        self.index: dict[str, int] = {}
        layers = []
        for name, (in_features, out_features) in named_shapes.items():
            self.index[name] = len(layers)
            layers.append(
                LoRALayer(
                    in_features=in_features,
                    out_features=out_features,
                    rank=LORA3_RANK,
                    alpha=LORA3_ALPHA,
                    dropout=LORA3_DROPOUT,
                )
            )
        self.layers = nn.ModuleList(layers)

    def delta(self, name: str, x: torch.Tensor) -> torch.Tensor:
        return self.layers[self.index[name]](x)


def _lora3_sites(model: nn.Module) -> dict[str, LoRALinear]:
    """视觉主干里 basename 恰好是 qkv 或 proj 的模态 LoRA。"""
    sites = {}
    for name, module in model.named_modules():
        if "vision_backbone" not in name:
            continue
        if name.split(".")[-1] not in {"qkv", "proj"}:
            continue
        if isinstance(module, LoRALinear):
            sites[name] = module
    return sites


def build_shared_lora(rgb_model: nn.Module, t_model: nn.Module) -> SharedLoRA:
    """在两套已注入模态 LoRA 的模型上挂同一套 LoRA3。不注册进 SAM3。"""
    rgb_sites = _lora3_sites(rgb_model)
    t_sites = _lora3_sites(t_model)
    if set(rgb_sites) != set(t_sites):
        raise RuntimeError("RGB 与 T 的 LoRA3 插入位置不一致。")
    if len(rgb_sites) != LORA3_LAYER_COUNT:
        raise RuntimeError(
            f"每套模型应有 {LORA3_LAYER_COUNT} 个视觉 qkv/proj LoRA，收到 {len(rgb_sites)}。"
        )

    named_shapes: dict[str, tuple[int, int]] = {}
    for name in sorted(rgb_sites):
        rgb_layer = rgb_sites[name]
        t_layer = t_sites[name]
        rgb_shape = (rgb_layer.in_features, rgb_layer.out_features)
        t_shape = (t_layer.in_features, t_layer.out_features)
        if rgb_shape != t_shape:
            raise RuntimeError(f"{name} 两侧线性层形状不一致：{rgb_shape} 与 {t_shape}。")
        named_shapes[name] = rgb_shape

    shared = SharedLoRA(named_shapes)
    for name in named_shapes:
        object.__setattr__(rgb_sites[name], "shared_lora", shared)
        object.__setattr__(rgb_sites[name], "shared_name", name)
        object.__setattr__(t_sites[name], "shared_lora", shared)
        object.__setattr__(t_sites[name], "shared_name", name)
    return shared


def enable_trunk_checkpoint(model: nn.Module) -> None:
    """打开 ViT trunk 的 activation checkpoint，并关掉 DropPath。"""
    trunk = model.backbone.vision_backbone.trunk
    trunk.train()
    for module in trunk.modules():
        if module.__class__.__name__ == "DropPath":
            module.eval()


def count_trainable(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def require_trainable(m1: nn.Module) -> list:
    params = [param for param in m1.parameters() if param.requires_grad]
    if not params:
        message = "M1 没有可训练参数。"
        logging.error(message)
        raise RuntimeError(message)
    return params


def make_find_stage(device: torch.device) -> FindStage:
    return FindStage(
        img_ids=torch.tensor([0], device=device, dtype=torch.long),
        text_ids=torch.tensor([0], device=device, dtype=torch.long),
        input_boxes=None,
        input_boxes_mask=None,
        input_boxes_label=None,
        input_points=None,
        input_points_mask=None,
    )


def make_geometric_prompt(boxes_cxcywh: torch.Tensor) -> Prompt:
    """正框，归一化 cxcywh，形状 (N, 1, 4)。标签和 padding mask 由 Prompt 补齐。"""
    if boxes_cxcywh.dim() != 2 or boxes_cxcywh.shape[-1] != 4:
        raise RuntimeError(f"框形状应为 (N, 4)，收到 {tuple(boxes_cxcywh.shape)}")
    count = boxes_cxcywh.shape[0]
    if count < 1:
        raise RuntimeError("至少需要一个框才能调用 Decoder。")
    boxes = boxes_cxcywh.to(dtype=torch.float32).view(count, 1, 4)
    return Prompt(box_embeddings=boxes)


def branch_backbone(fused: dict, text_outputs: dict) -> dict:
    """复制融合特征和文本。文本在整个训练里复用，不能被 Decoder 原地改写。"""
    features = clone_tree(fused)
    for key, value in text_outputs.items():
        features[key] = clone_tree(value)
    return features


def decode(model: nn.Module, features: dict, boxes_cxcywh: torch.Tensor, find_stage: FindStage) -> dict:
    if model.training:
        raise RuntimeError("SAM3 必须保持 eval()。")
    return model.forward_grounding(
        backbone_out=features,
        find_input=find_stage,
        find_target=None,
        geometric_prompt=make_geometric_prompt(boxes_cxcywh),
    )


def merge_topk_logits(outputs: dict, top_k: int) -> torch.Tensor:
    """按 presence 调整后的分类分数取 top-k，对 mask logits 做逐像素 max。"""
    if top_k < 1:
        raise ValueError(f"top_k 必须 >= 1，收到 {top_k}")
    pred_logits = outputs["pred_logits"]
    pred_masks = outputs["pred_masks"]
    if pred_logits.dim() != 3 or pred_masks.dim() != 4:
        raise RuntimeError(
            "Decoder 输出形状异常: "
            f"pred_logits {tuple(pred_logits.shape)}, pred_masks {tuple(pred_masks.shape)}"
        )

    scores = pred_logits.sigmoid()
    presence = outputs.get("presence_logit_dec")
    if presence is not None:
        scores = scores * presence.sigmoid().unsqueeze(1)
    query_scores = scores[0].squeeze(-1)
    query_masks = pred_masks[0]
    k = min(int(top_k), int(query_scores.numel()))
    index = torch.topk(query_scores, k=k).indices
    return query_masks.index_select(0, index).max(dim=0).values


def upsample_logits(logit: torch.Tensor, size: tuple) -> torch.Tensor:
    return F.interpolate(logit.float()[None, None], size=size, mode="bilinear", align_corners=False)


def encode_visual_text(model: nn.Module, device: torch.device) -> dict:
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        return model.backbone.forward_text(["visual"], device=device)


def forward_sample(
    rgb: torch.Tensor,
    thermal: torch.Tensor,
    gt_boxes: torch.Tensor,
    rgb_model: nn.Module,
    t_model: nn.Module,
    m1: M1,
    text_outputs: dict,
    find_stage: FindStage,
    grad_backbone: bool = False,
) -> torch.Tensor:
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        if grad_backbone:
            rgb_out = rgb_model.backbone.forward_image(rgb)
            t_out = t_model.backbone.forward_image(thermal)
        else:
            with torch.no_grad():
                rgb_out = rgb_model.backbone.forward_image(rgb)
                t_out = t_model.backbone.forward_image(thermal)
        fused = m1(rgb_out, t_out)
        outputs = decode(rgb_model, branch_backbone(fused, text_outputs), gt_boxes, find_stage)
    return merge_topk_logits(outputs, int(gt_boxes.shape[0]))


def resize_gt(gt_mask: torch.Tensor, size: tuple) -> torch.Tensor:
    """最近邻把 (H, W) 的 GT 缩放到训练分辨率，保持二值。"""
    return F.interpolate(gt_mask.float()[None, None], size=size, mode="nearest")


def sigmoid_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = FOCAL_ALPHA,
    gamma: float = FOCAL_GAMMA,
) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = prob * target + (1.0 - prob) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    return (alpha_t * (1.0 - p_t).pow(gamma) * ce).mean()


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = tuple(range(1, prob.dim()))
    inter = (prob * target).sum(dim=dims)
    denom = prob.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def iou_counts(pred: torch.Tensor, target: torch.Tensor) -> tuple[int, int, int]:
    """与 evaluate2.AS1000Metrics.update 相同的像素 TP / FP / FN。"""
    pred = pred.detach().reshape(-1) > 0
    target = target.detach().reshape(-1).to(device=pred.device) > 0
    tp = pred & target
    fp = pred & ~target
    fn = ~pred & target
    return int(tp.sum().item()), int(fp.sum().item()), int(fn.sum().item())


def dataset_iou(tp: int, fp: int, fn: int, eps: float = 1e-10) -> float:
    """与 evaluate2.AS1000Metrics.compute 的 IoU 相同，返回 0 到 1 的原始比例。"""
    return tp / (tp + fp + fn + eps)


def format_metric(value: float, places: int = 5) -> str:
    """四舍五入到指定小数位。"""
    quant = Decimal("1").scaleb(-places)
    number = Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP)
    return f"{number:.{places}f}"


def prediction_probability(logit: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
    """预测概率图缩放到 GT 分辨率，数值限制在 [0, 1]。"""
    probability = torch.sigmoid(logit.float())[None, None]
    probability = F.interpolate(probability, size=tuple(gt_mask.shape[-2:]), mode="bilinear", align_corners=False)[0, 0]
    return probability.clamp(0.0, 1.0)


def train_one_epoch(
    loader,
    rgb_model,
    t_model,
    m1,
    lora3,
    text_outputs,
    find_stage,
    optimizer,
    epoch: int,
    args: argparse.Namespace,
) -> float:
    rgb_model.eval()
    t_model.eval()
    m1.train()
    if lora3 is not None:
        lora3.train()
        enable_trunk_checkpoint(rgb_model)
        enable_trunk_checkpoint(t_model)

    total_loss = 0.0
    total_step = len(loader)
    for step, batch in enumerate(loader, start=1):
        optimizer.zero_grad(set_to_none=True)
        rgb = batch["rgb"].unsqueeze(0).cuda(non_blocking=True)
        thermal = batch["t"].unsqueeze(0).cuda(non_blocking=True)
        gt_boxes = batch["boxes"].cuda(non_blocking=True)
        gt_mask = batch["gt_mask"].cuda(non_blocking=True)

        pred_logit = forward_sample(
            rgb,
            thermal,
            gt_boxes,
            rgb_model,
            t_model,
            m1,
            text_outputs,
            find_stage,
            grad_backbone=lora3 is not None,
        )
        pred_map = upsample_logits(pred_logit, rgb.shape[-2:])
        gt = resize_gt(gt_mask, rgb.shape[-2:])

        str_term = args.str_weight * structure_loss(pred_map, gt)
        focal_term = args.focal_weight * sigmoid_focal_loss(pred_map, gt)
        dice_term = args.dice_weight * dice_loss(pred_map, gt)
        loss = str_term + focal_term + dice_term
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        if step == 1 or step == total_step or step % args.log_every == 0:
            message = (
                f"{datetime.now()} Epoch [{epoch:03d}/{args.epoch:03d}], "
                f"Step [{step:04d}/{total_step:04d}], "
                f"str: {str_term.item():.6f}, "
                f"focal: {focal_term.item():.6f}, "
                f"dice: {dice_term.item():.6f}, "
                f"total: {loss.item():.6f}"
            )
            print(message)
            logging.info("#TRAIN# %s", message)

    return total_loss / max(total_step, 1)


@torch.no_grad()
def validate(loader, rgb_model, t_model, m1, lora3, text_outputs, find_stage) -> dict:
    rgb_model.eval()
    t_model.eval()
    m1.eval()
    if lora3 is not None:
        lora3.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_mae = 0.0
    count = 0
    for batch in loader:
        rgb = batch["rgb"].unsqueeze(0).cuda(non_blocking=True)
        thermal = batch["t"].unsqueeze(0).cuda(non_blocking=True)
        gt_boxes = batch["boxes"].cuda(non_blocking=True)
        gt_mask = batch["gt_mask"]
        orig_h, orig_w = batch["orig_size"]
        if tuple(gt_mask.shape[-2:]) != (orig_h, orig_w):
            raise RuntimeError(
                f"{batch['stem']} 的 GT 尺寸 {tuple(gt_mask.shape[-2:])} 与 RGB 尺寸 {(orig_h, orig_w)} 不一致"
            )

        pred_logit = forward_sample(
            rgb, thermal, gt_boxes, rgb_model, t_model, m1, text_outputs, find_stage
        )
        probability = prediction_probability(pred_logit, gt_mask)
        tp, fp, fn = iou_counts(probability > 0.5, gt_mask)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_mae += calc_mae(probability, gt_mask)
        count += 1

    scale = max(count, 1)
    return {"iou": dataset_iou(total_tp, total_fp, total_fn), "mae": total_mae / scale, "samples": count}


def save_checkpoint(path: Path, m1: M1, epoch: int, iou: float, lora3: nn.Module | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "m1": m1.state_dict(),
            "lora3": None if lora3 is None else lora3.state_dict(),
            "epoch": epoch,
            "iou": iou,
        },
        path,
    )


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="全监督 RGB-T 训练：冻结 SAM3 与双 LoRA，训练 M1 与共享 LoRA3")
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batchsize", type=int, default=1)
    parser.add_argument("--img_size", type=int, default=1008)
    parser.add_argument("--decay_rate", type=float, default=0.1)
    parser.add_argument("--decay_epoch", type=int, default=50)
    parser.add_argument("--str_weight", type=float, default=1.0)
    parser.add_argument("--focal_weight", type=float, default=1.0)
    parser.add_argument("--dice_weight", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=100)

    parser.add_argument("--lora3", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--checkpoint", type=str, default="/share_datasets/Seal_Model_Reproduction/Ours/weights/sam3.pt")
    parser.add_argument("--rgb_lora", type=str, default="/share_datasets/Seal_Model_Reproduction/Ours/weights/RGB+LoRA/lora_weights.pt")
    parser.add_argument("--t_lora", type=str, default="/share_datasets/Seal_Model_Reproduction/Ours/weights/T+LoRA/lora_weights.pt")
    parser.add_argument("--train_data_dir", type=str, default="/share_datasets/Seal_Model_Reproduction/Seal_data/train")
    parser.add_argument("--val_data_dir", type=str, default="/share_datasets/Seal_Model_Reproduction/Seal_data/val")
    parser.add_argument("--save_path", type=str, default="train_logs")
    return parser.parse_args()


def main() -> None:
    args = get_args()
    setup_logger(args.save_path, "train.log")
    logging.info("Args: %s", args)
    print("Learning Rate:", args.lr)

    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA。")
    device = torch.device("cuda")

    checkpoint = Path(args.checkpoint)
    rgb_model = build_frozen_lora_model(checkpoint, Path(args.rgb_lora), device)
    t_model = build_frozen_lora_model(checkpoint, Path(args.t_lora), device)
    lora3 = build_shared_lora(rgb_model, t_model).to(device) if args.lora3 else None
    m1 = M1().cuda()
    m1_trainable = count_trainable(m1)
    lora3_trainable = 0 if lora3 is None else count_trainable(lora3)
    rgb_trainable = count_trainable(rgb_model)
    t_trainable = count_trainable(t_model)
    logging.info(
        "可训练参数 m1=%d lora3=%d，冻结后 rgb=%d t=%d",
        m1_trainable,
        lora3_trainable,
        rgb_trainable,
        t_trainable,
    )
    if rgb_trainable or t_trainable:
        raise RuntimeError("SAM3 与模态 LoRA 应当全部冻结。")
    params = require_trainable(m1)
    if lora3 is not None:
        lora3_params = [param for param in lora3.parameters() if param.requires_grad]
        if not lora3_params:
            raise RuntimeError("LoRA3 没有可训练参数。")
        params.extend(lora3_params)

    train_loader = get_loader(
        args.train_data_dir,
        image_size=args.img_size,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.num_workers,
        load_gt=True,
    )
    val_loader = get_loader(
        args.val_data_dir,
        image_size=args.img_size,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.num_workers,
        load_gt=True,
    )
    text_outputs = encode_visual_text(rgb_model, device)
    find_stage = make_find_stage(device)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    best_iou = -1.0
    best_epoch = 0
    checkpoint_path = Path(args.save_path) / "best.pth"
    print("Go! Go! Go!")
    for epoch in range(1, args.epoch + 1):
        lr = adjust_lr(optimizer, args.lr, epoch, decay_rate=args.decay_rate, decay_epoch=args.decay_epoch)
        logging.info("Epoch %d lr %.8f", epoch, lr)
        train_loss = train_one_epoch(
            train_loader,
            rgb_model,
            t_model,
            m1,
            lora3,
            text_outputs,
            find_stage,
            optimizer,
            epoch,
            args,
        )
        logging.info("#TRAIN# epoch=%d mean_loss=%.6f", epoch, train_loss)

        metrics = validate(val_loader, rgb_model, t_model, m1, lora3, text_outputs, find_stage)
        if metrics["samples"] > 0 and metrics["iou"] > best_iou:
            best_iou = metrics["iou"]
            best_epoch = epoch
            save_checkpoint(checkpoint_path, m1, epoch, metrics["iou"], lora3)
            logging.info("保存最佳权重 %s", checkpoint_path)

        message = (
            f"Epoch: {epoch} IoU: {format_metric(metrics['iou'])} MAE: {metrics['mae']:.6f} "
            f"bestIoU: {format_metric(best_iou)} bestEpoch: {best_epoch} samples: {metrics['samples']}"
        )
        print(message)
        logging.info("#VAL# %s", message)

    logging.info("训练结束。最佳验证 IoU %s，出现在 epoch %d", format_metric(best_iou), best_epoch)


if __name__ == "__main__":
    main()
