from __future__ import annotations
import argparse
import logging
import math
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from math import log
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union
from PIL import Image
from iopath.common.file_io import g_pathmgr
from sam3.adapter import DualLoRALinear


EXT_PRIORITY = [".png", ".jpg", ".jpeg", ".bmp"]

_LAPLACIAN_3X3 = torch.tensor(
    [[-1, -1, -1],
     [-1, 8, -1],
     [-1, -1, -1]],
    dtype=torch.float32,
)

_SOBEL_X_3X3 = torch.tensor(
    [[-1, 0, 1],
     [-2, 0, 2],
     [-1, 0, 1]],
    dtype=torch.float32,
)

_SOBEL_Y_3X3 = torch.tensor(
    [[-1, -2, -1],
     [0, 0, 0],
     [1, 2, 1]],
    dtype=torch.float32,
)


class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias),
            nn.GroupNorm(num_groups=min(32, planes), num_channels=planes),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)


# [DISABLED:GEA] class GEA(nn.Module):
# [DISABLED:GEA]     """梯度边缘聚合模块 - 参考EG-SAM设计
# [DISABLED:GEA]     输入：浅层特征(B, 32, 288, 288) + 深层特征(B, 256, 72, 72)
# [DISABLED:GEA]     输出：边缘logits(B, 1, 288, 288)
# [DISABLED:GEA]     """
# [DISABLED:GEA]     def __init__(self, shallow_dim=32, deep_dim=256, embed_dim=64):
# [DISABLED:GEA]         super(GEA, self).__init__()
# [DISABLED:GEA]         self.shallow_dim = shallow_dim
# [DISABLED:GEA]         self.deep_dim = deep_dim
# [DISABLED:GEA]         self.embed_dim = embed_dim
#
# [DISABLED:GEA]         self.reduce_shallow = ConvBNR(shallow_dim, embed_dim, kernel_size=1)
# [DISABLED:GEA]         self.reduce_deep = ConvBNR(deep_dim, embed_dim, kernel_size=1)
#
# [DISABLED:GEA]         # 融合网络：融合浅层梯度和深层语义
# [DISABLED:GEA]         self.edge_block = nn.Sequential(
# [DISABLED:GEA]             ConvBNR(embed_dim * 2, embed_dim, kernel_size=3, stride=1, dilation=1),
# [DISABLED:GEA]             ConvBNR(embed_dim, embed_dim, kernel_size=3, stride=1, dilation=1),
# [DISABLED:GEA]             nn.Conv2d(embed_dim, 1, kernel_size=1)
# [DISABLED:GEA]         )
#
# [DISABLED:GEA]         # 固定的Sobel算子
# [DISABLED:GEA]         self.register_buffer('sobel_x',
# [DISABLED:GEA]                              torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
# [DISABLED:GEA]         self.register_buffer('sobel_y',
# [DISABLED:GEA]                              torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))
#
# [DISABLED:GEA]         nn.init.constant_(self.edge_block[-1].bias, -5.0)
#
# [DISABLED:GEA]     def compute_gradient(self, x):
# [DISABLED:GEA]         """计算特征的梯度幅度"""
# [DISABLED:GEA]         x_gray = x.mean(dim=1, keepdim=True)
# [DISABLED:GEA]         gx = F.conv2d(x_gray, self.sobel_x, padding=1)
# [DISABLED:GEA]         gy = F.conv2d(x_gray, self.sobel_y, padding=1)
# [DISABLED:GEA]         return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
#
# [DISABLED:GEA]     def forward(self, shallow_feats, deep_feats):
# [DISABLED:GEA]         """
# [DISABLED:GEA]         Args:
# [DISABLED:GEA]             shallow_feats: (B, 32, 288, 288) - 浅层特征
# [DISABLED:GEA]             deep_feats: (B, 256, 72, 72) - 深层特征
# [DISABLED:GEA]         Returns:
# [DISABLED:GEA]             edge_logits: (B, 1, 288, 288) - 边缘logits
# [DISABLED:GEA]         """
# [DISABLED:GEA]         # 深层特征上采样到浅层分辨率
# [DISABLED:GEA]         deep_up = F.interpolate(deep_feats, size=shallow_feats.shape[-2:], mode='bilinear', align_corners=False)
#
# [DISABLED:GEA]         # 特征升维
# [DISABLED:GEA]         xs = self.reduce_shallow(shallow_feats)
# [DISABLED:GEA]         xd = self.reduce_deep(deep_up)
#
# [DISABLED:GEA]         # 计算浅层梯度
# [DISABLED:GEA]         s_grad = self.compute_gradient(xs)
# [DISABLED:GEA]         xs_edge = xs * s_grad
#
# [DISABLED:GEA]         # 融合梯度和语义信息
# [DISABLED:GEA]         feat = torch.cat([xs_edge, xd], dim=1)
# [DISABLED:GEA]         edge_logits = self.edge_block(feat)
#
# [DISABLED:GEA]         return edge_logits
#

# [DISABLED:EDM] class EDM(nn.Module):
# [DISABLED:EDM]     """Edge-Driven Module - modelled after EG-SAM EDM.
# [DISABLED:EDM]     Bi-directional fusion + ECA channel attention.
# [DISABLED:EDM]     ConvBNR is replaced by plain Conv+residual to preserve feature magnitude
# [DISABLED:EDM]     (BN would normalize away the enhancement from bi-directional fusion).
# [DISABLED:EDM]     """
# [DISABLED:EDM]     def __init__(self, channel=32):
# [DISABLED:EDM]         super(EDM, self).__init__()
# [DISABLED:EDM]         t = int(abs((math.log(channel, 2) + 1) / 2))
# [DISABLED:EDM]         k = t if t % 2 else t + 1
# [DISABLED:EDM]         # Plain conv (no BN/ReLU) to preserve feature magnitude
# [DISABLED:EDM]         self.conv2d_1 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
# [DISABLED:EDM]         self.conv2d_2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
# [DISABLED:EDM]         self.avg_pool = nn.AdaptiveAvgPool2d(1)
# [DISABLED:EDM]         self.conv1d   = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
# [DISABLED:EDM]         self.sigmoid  = nn.Sigmoid()
# [DISABLED:EDM]         # Initialize convs near identity to avoid disrupting features at start
# [DISABLED:EDM]         nn.init.normal_(self.conv2d_1.weight, mean=0, std=0.01)
# [DISABLED:EDM]         nn.init.zeros_(self.conv2d_1.bias)
# [DISABLED:EDM]         nn.init.normal_(self.conv2d_2.weight, mean=0, std=0.01)
# [DISABLED:EDM]         nn.init.zeros_(self.conv2d_2.bias)
#
# [DISABLED:EDM]     def forward(self, edge_logits, hq_feature):
# [DISABLED:EDM]         """
# [DISABLED:EDM]         Args:
# [DISABLED:EDM]             edge_logits: (B, 1, 288, 288) - GEA output edge logits
# [DISABLED:EDM]             hq_feature:  (B, 32, 288, 288) - shallow high-res feature
# [DISABLED:EDM]         Returns:
# [DISABLED:EDM]             enhanced_feat: (B, 32, 288, 288) - edge-enhanced feature
# [DISABLED:EDM]         """
# [DISABLED:EDM]         edge_prob = torch.sigmoid(edge_logits)  # (B, 1, 288, 288), range [0,1]
#
# [DISABLED:EDM]         # Bi-directional fusion (EG-SAM style):
# [DISABLED:EDM]         #   x = hq*e + hq  (edge-weighted feature + feature residual)
# [DISABLED:EDM]         #   y = e*hq + e   (feature-weighted edge + edge residual)
# [DISABLED:EDM]         #   fused = x + y = hq*(1+2e) + e
# [DISABLED:EDM]         x = hq_feature * edge_prob + hq_feature
# [DISABLED:EDM]         y = edge_prob * hq_feature + edge_prob
# [DISABLED:EDM]         fused = x + y
#
# [DISABLED:EDM]         # Learnable refinement with residual (preserves magnitude)
# [DISABLED:EDM]         x = self.conv2d_1(fused)
# [DISABLED:EDM]         x = self.conv2d_2(x)
# [DISABLED:EDM]         x = x + fused  # residual: near-identity at init
#
# [DISABLED:EDM]         # ECA channel attention
# [DISABLED:EDM]         wei = self.avg_pool(x)
# [DISABLED:EDM]         wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
# [DISABLED:EDM]         wei = self.sigmoid(wei) * 0.3 + 0.7  # range [0.7, 1.0]: attenuate not zero-out
# [DISABLED:EDM]         x = x * wei
#
# [DISABLED:EDM]         return x
#
#

def find_vit_blocks(backbone: nn.Module) -> nn.ModuleList:
    """递归或属性查找 ViT blocks，辅助 inject_dual_lora 使用"""
    candidates = [
        ("vision_backbone", "trunk", "blocks"),
        ("visual", "trunk", "blocks"),
        ("trunk", "blocks"),
        ("blocks",)
    ]

    for path in candidates:
        node = backbone
        try:
            for attr in path:
                node = getattr(node, attr)
            return node
        except AttributeError:
            continue

    raise AttributeError(f"Cannot locate trunk.blocks in {type(backbone)}")


def inject_dual_lora(backbone: nn.Module) -> nn.ModuleList:
    """在 SAM3 ViT 图像编码器所有 block 的 QKV 上注入 DualLoRALinear。"""
    dual_qkvs = nn.ModuleList()
    vit_blocks = find_vit_blocks(backbone)

    for blk in vit_blocks:
        if not (hasattr(blk, "attn") and hasattr(blk.attn, "qkv")):
            continue

        original_qkv = blk.attn.qkv

        if isinstance(original_qkv, DualLoRALinear):
            dual_qkvs.append(original_qkv)
            continue

        dual_lora = DualLoRALinear(original_qkv, mode="rgb")
        blk.attn.qkv = dual_lora
        dual_qkvs.append(dual_lora)

    if len(dual_qkvs) == 0:
        raise RuntimeError("No QKV layers found for LoRA injection in SAM3 backbone")

    return dual_qkvs


def load_tracker_checkpoint(tracker: nn.Module, path: str) -> None:
    if not g_pathmgr.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")

    state = obj.get("model", obj) if isinstance(obj, dict) else obj
    if not isinstance(state, dict):
        raise ValueError("Checkpoint is not a valid state dict")

    mapped = {}
    for k, v in state.items():
        if not isinstance(v, torch.Tensor):
            continue
        if k.startswith("tracker."):
            mapped[k.replace("tracker.", "", 1)] = v
        elif k.startswith("detector.backbone.") and "language" not in k and ".text." not in k:
            mapped["backbone." + k[len("detector.backbone."):]] = v

    if not mapped:
        raise RuntimeError("Checkpoint contains no usable tracker weights")

    tracker.load_state_dict(mapped, strict=False)


def load_finetuned_checkpoint(model: nn.Module, path: str) -> None:
    """加载微调后的权重，仅保留 Tensor 类型"""
    if not g_pathmgr.exists(path):
        raise FileNotFoundError(f"Finetune Checkpoint not found: {path}")

    obj = torch.load(path, map_location="cpu")
    state = obj.get("model", obj) if isinstance(obj, dict) else obj

    state = {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}

    model.load_state_dict(state, strict=False)


def reshape_sam3_features(features: List[torch.Tensor], sizes: List[Tuple[int, int]]) -> List[torch.Tensor]:
    """将 SAM3 Backbone 输出的 (HW, B, C) 特征重塑为 (B, C, H, W)"""
    reshaped_feats = []
    for f, s in zip(features, sizes):
        # f shape: [HW, B, C] -> permute(1, 2, 0) -> [B, C, HW] -> view -> [B, C, H, W]
        f_reshaped = f.permute(1, 2, 0).contiguous().view(f.shape[1], -1, *s)
        reshaped_feats.append(f_reshaped)
    return reshaped_feats


def resolve_output_size(out_hw: Optional[torch.Tensor], default: int) -> Tuple[int, int]:
    """解析输出分辨率"""
    if out_hw is None:
        return default, default

    flat = out_hw.view(-1)
    if flat.numel() >= 2:
        return int(flat[0].item()), int(flat[1].item())
    return default, default


def list_images_unique_by_stem(root: Union[str, Path], *, exts: Iterable[str], priority: Optional[Sequence[str]] = None) -> Dict[str, Path]:
    """扫描目录，根据文件名主干去重"""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Path not found: {root}")

    if priority is None:
        priority = EXT_PRIORITY

    exts_set = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}
    prio_map = {ext.lower(): i for i, ext in enumerate(priority)}
    default_rank = len(priority)

    best_files: Dict[str, Path] = {}

    for p in root.iterdir():
        if not p.is_file():
            continue

        suffix = p.suffix.lower()
        if suffix not in exts_set:
            continue

        stem = p.stem
        rank = prio_map.get(suffix, default_rank)

        if stem not in best_files:
            best_files[stem] = p
        else:
            current_rank = prio_map.get(best_files[stem].suffix.lower(), default_rank)
            if rank < current_rank:
                best_files[stem] = p

    return best_files


def align_modalities_by_stem_strict(maps: Dict[str, Dict[str, Path]]) -> Dict[str, List[Path]]:
    """严格对齐不同模态的文件，确保 Stem 集合一致"""
    if not maps:
        raise ValueError("No modality maps provided.")

    stem_sets = {k: set(m.keys()) for k, m in maps.items()}
    common_stems = set.intersection(*stem_sets.values())

    if not common_stems:
        raise ValueError(f"No common stems found among modalities: {list(maps.keys())}")

    for k, s_set in stem_sets.items():
        if s_set != common_stems:
            missing = common_stems - s_set
            extra = s_set - common_stems
            err_msg = [f"Modality alignment failed for '{k}':"]
            if missing:
                err_msg.append(f"  Missing ({len(missing)}): {list(missing)[:5]}...")
            if extra:
                err_msg.append(f"  Extra ({len(extra)}): {list(extra)[:5]}...")
            raise ValueError("\n".join(err_msg))

    sorted_stems = sorted(common_stems)
    return {k: [maps[k][s] for s in sorted_stems] for k in maps}


def permuted_kfold_indices(total: int, *, k_folds: int, fold_idx: int, seed: int) -> Tuple[List[int], List[int]]:
    """生成 K-Fold 的训练和验证索引"""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(total).tolist()

    fold_size = total // k_folds
    val_start = fold_idx * fold_size
    val_end = (fold_idx + 1) * fold_size if fold_idx < k_folds - 1 else total

    val_indices = perm[val_start:val_end]
    train_indices = perm[:val_start] + perm[val_end:]
    return train_indices, val_indices


def open_rgb(path: Path) -> Image.Image:
    with path.open("rb") as f:
        return Image.open(f).convert("RGB")


def open_gray(path: Path) -> Image.Image:
    with path.open("rb") as f:
        return Image.open(f).convert("L")


def open_thermal(path: Path) -> Image.Image:
    """16 位热红外 TIFF：按单图 min-max 拉伸到 8 位，再转三通道 RGB。

    Seal 的 ``T`` 是 16 位，直接 ``convert("RGB")`` 会把绝大多数像素截到 255；
    全项目统一用逐图 min-max 映射到 8 位（见 AGENTS.md）。
    """
    with path.open("rb") as f:
        array = np.array(Image.open(f))
    if array.ndim == 3:
        array = array[..., 0]
    array = array.astype(np.float32)
    low, high = float(array.min()), float(array.max())
    if high > low:
        array = (array - low) * (255.0 / (high - low))
    else:
        array = np.zeros_like(array)
    array = np.clip(np.rint(array), 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(array, mode="L").convert("RGB")


def read_bbox_txt(path: Path) -> torch.Tensor:
    """读取 bbox txt 文件，支持单框或多框，每个框格式: x_min y_min x_max y_max"""
    with open(path, 'r') as f:
        content = f.read().strip().split()
        coords = [float(x) for x in content]

    if len(coords) == 0 or len(coords) % 4 != 0:
        raise ValueError(f"Invalid bbox format in {path}. Expected a multiple of 4 values, got {len(coords)}")

    return torch.tensor(coords, dtype=torch.float32).view(-1, 4)


def sample_prompt_points(gt_map: Image.Image, *, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算前景涂鸦的 BBox，将其划分为 3x3 九宫格，从每个包含前景的格子中随机采样 1 个点"""
    rng = np.random.RandomState(seed)
    gt = np.array(gt_map)

    fg_indices = np.argwhere(gt == 255)

    if len(fg_indices) == 0:
        return torch.empty((0, 2)), torch.empty((0,), dtype=torch.int64)

    y_min, x_min = fg_indices.min(axis=0)
    y_max, x_max = fg_indices.max(axis=0)

    step_y = (y_max - y_min) / 3.0
    step_x = (x_max - x_min) / 3.0

    points_list = []

    for i in range(3):
        for j in range(3):
            y_start = y_min + i * step_y
            y_end = y_min + (i + 1) * step_y if i < 2 else y_max + 1
            x_start = x_min + j * step_x
            x_end = x_min + (j + 1) * step_x if j < 2 else x_max + 1

            in_grid = (
                    (fg_indices[:, 0] >= y_start) & (fg_indices[:, 0] < y_end) &
                    (fg_indices[:, 1] >= x_start) & (fg_indices[:, 1] < x_end)
            )

            valid_indices = fg_indices[in_grid]

            if len(valid_indices) > 0:
                idx = rng.choice(len(valid_indices))
                points_list.append(valid_indices[idx][::-1])

    points_tensor = torch.from_numpy(np.array(points_list, dtype=np.float32))
    labels_tensor = torch.ones(len(points_list), dtype=torch.int64)

    return points_tensor, labels_tensor


def sample_one_point_from_scribble(gt_map: Image.Image, *, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """从涂鸦中随机抽取 1 个前景点"""
    rng = np.random.RandomState(seed)
    gt = np.array(gt_map)

    fg_indices = np.argwhere(gt == 255)

    points_list, labels_list = [], []

    if fg_indices.size > 0:
        idx = rng.randint(len(fg_indices))
        points_list.append(fg_indices[idx][None, [1, 0]])  # (y, x) -> (x, y)
        labels_list.append(np.array([1], dtype=np.int64))

    return torch.from_numpy(np.concatenate(points_list, axis=0)), torch.from_numpy(np.concatenate(labels_list, axis=0))


def preprocess_sam3_image(x: torch.Tensor, *, image_size: int, mean: float, std: float) -> torch.Tensor:
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    else:
        x = x.float()
        if x.max() > 1.1:
            x = x / 255.0

    x = torch.clamp(x, 0.0, 1.0)

    if x.dim() == 2:
        x = x.unsqueeze(0)
    if x.shape[0] == 1:
        x = x.expand(3, -1, -1)

    if x.shape[-2:] != (image_size, image_size):
        x = F.interpolate(x.unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False).squeeze(0)

    return (x - mean) / std


def scale_points(point_xy: torch.Tensor, *, in_hw: Tuple[int, int], out_size: int) -> torch.Tensor:
    """将坐标从原始尺寸缩放到 SAM3 输入尺寸"""
    if point_xy.numel() == 0:
        return point_xy

    h, w = in_hw
    scale_x = out_size / float(w)
    scale_y = out_size / float(h)

    out = point_xy.float().clone()
    out[:, 0] *= scale_x
    out[:, 1] *= scale_y
    return out


def setup_logger(save_path: str, filename: str = 'log.txt') -> None:
    Path(save_path).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(save_path, filename),
        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
        level=logging.INFO,
        filemode='a',
        datefmt='%Y-%m-%d %I:%M:%S %p',
        force=True,
    )


def adjust_lr(optimizer: torch.optim.Optimizer, init_lr: float, epoch: int, *, decay_rate: float = 0.1, decay_epoch: int = 30,) -> float:
    """
    Step decay learning rate:
        lr = init_lr * decay_rate ** (epoch // decay_epoch)

    Returns the new lr (for convenience / logging).
    """
    step = epoch // decay_epoch
    lr = init_lr * (decay_rate ** step)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def to_device(data: Any, device: torch.device) -> Any:
    """递归地将 Tensor、List、Tuple 或 Dict 移动到指定设备"""
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True)
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(x, device) for x in data)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    return data


def load_offline_edge(base_dir: Union[str, Path], name: str, device: torch.device) -> torch.Tensor:
    """按需加载离线生成的 Canny 边缘图"""
    base_dir = Path(base_dir)
    edge_path = base_dir / f"{name}.png"

    if not edge_path.exists():
        raise FileNotFoundError(f"Missing offline Canny edge map for {name} at {edge_path}")

    edge_img = Image.open(edge_path).convert('L')

    edge_np = np.array(edge_img, dtype=np.float32)

    edge_tensor = torch.from_numpy(edge_np) / 255.0

    return edge_tensor.unsqueeze(0).unsqueeze(0).to(device)


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))


def charbonnier(x: torch.Tensor, *, eps: float = 1e-3) -> torch.Tensor:
    # sqrt(x^2 + eps^2)
    return torch.sqrt(x.square() + (eps * eps))


@dataclass(frozen=True)
class SmoothnessConfig:
    alpha: float = 10.0
    s1: float = 10.0
    s2: float = 1.0
    eps: float = 1e-3
    reduction: str = "mean"  # "mean" or "sum"


class SmoothnessLoss(torch.nn.Module):
    def __init__(self, cfg: Optional[SmoothnessConfig] = None):
        super().__init__()
        self.cfg = cfg or SmoothnessConfig()

        self.register_buffer('laplacian_kernel', _LAPLACIAN_3X3.view(1, 1, 3, 3))
        self.register_buffer('sobel_x_kernel', _SOBEL_X_3X3.view(1, 1, 3, 3))
        self.register_buffer('sobel_y_kernel', _SOBEL_Y_3X3.view(1, 1, 3, 3))

    def _conv2d(self, img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        return F.conv2d(img, kernel, stride=1, padding=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")

        sal_x = self._conv2d(pred, self.sobel_x_kernel).abs()
        sal_y = self._conv2d(pred, self.sobel_y_kernel).abs()
        gt_x = self._conv2d(target, self.sobel_x_kernel)
        gt_y = self._conv2d(target, self.sobel_y_kernel)

        w_x = torch.exp(-self.cfg.alpha * gt_x.abs())
        w_y = torch.exp(-self.cfg.alpha * gt_y.abs())

        cps = charbonnier(sal_x * w_x, eps=self.cfg.eps) + charbonnier(sal_y * w_y, eps=self.cfg.eps)

        lap_sal = self._conv2d(pred, self.laplacian_kernel).abs()
        lap_gt = self._conv2d(target, self.laplacian_kernel).abs()

        w_lap = torch.exp(-self.cfg.alpha * lap_gt)
        lap_term = charbonnier(lap_sal * w_lap, eps=self.cfg.eps)

        if self.cfg.reduction == "mean":
            return self.cfg.s1 * torch.mean(cps) + self.cfg.s2 * torch.mean(lap_term)
        elif self.cfg.reduction == "sum":
            return self.cfg.s1 * torch.sum(cps) + self.cfg.s2 * torch.sum(lap_term)
        else:
            raise ValueError(f"Unsupported reduction: {self.cfg.reduction}")


class LocalSaliencyCoherence(torch.nn.Module):
    """Gated CRF Loss"""
    def forward(
            self, y_hat_softmax, kernels_desc, kernels_radius, sample, height_input, width_input,
            mask_src=None, mask_dst=None, compatibility=None, custom_modality_downsamplers=None, out_kernels_vis=False
    ):
        assert y_hat_softmax.dim() == 4, 'Prediction must be a NCHW batch'
        N, C, height_pred, width_pred = y_hat_softmax.shape

        device = y_hat_softmax.device

        assert width_input % width_pred == 0 and height_input % height_pred == 0 and \
               width_input * height_pred == height_input * width_pred, \
            f'[{width_input}x{height_input}] !~= [{width_pred}x{height_pred}]'

        kernels = self._create_kernels(kernels_desc, kernels_radius, sample, N, height_pred, width_pred, device, custom_modality_downsamplers)

        y_hat_unfolded = self._unfold(y_hat_softmax, kernels_radius)
        y_hat_unfolded = torch.abs(y_hat_unfolded[:, :, kernels_radius, kernels_radius, :, :].view(N, C, 1, 1, height_pred, width_pred) - y_hat_unfolded)

        loss = torch.mean((kernels * y_hat_unfolded).view(N, C, (kernels_radius * 2 + 1) ** 2, height_pred, width_pred).sum(dim=2, keepdim=True))

        out = {'loss': loss.mean(),}

        if out_kernels_vis:
            out['kernels_vis'] = self._visualize_kernels(kernels, kernels_radius, height_input, width_input, height_pred, width_pred)

        return out

    @staticmethod
    def _downsample(img, modality, height_dst, width_dst, custom_modality_downsamplers):
        if custom_modality_downsamplers is not None and modality in custom_modality_downsamplers:
            f_down = custom_modality_downsamplers[modality]
        else:
            f_down = F.adaptive_avg_pool2d
        return f_down(img, (height_dst, width_dst))

    @staticmethod
    def _create_kernels(kernels_desc, kernels_radius, sample, N, height_pred, width_pred, device, custom_modality_downsamplers):
        kernels = None
        for i, desc in enumerate(kernels_desc):
            weight = desc['weight']
            features = []
            for modality, sigma in desc.items():
                if modality == 'weight':
                    continue
                if modality == 'xy':
                    feature = LocalSaliencyCoherence._get_mesh(N, height_pred, width_pred, device)
                else:
                    assert modality in sample, f'Modality {modality} is listed in {i}-th kernel descriptor, but not present in the sample'
                    feature = sample[modality]
                feature /= sigma
                features.append(feature)
            features = torch.cat(features, dim=1)
            kernel = weight * LocalSaliencyCoherence._create_kernels_from_features(features, kernels_radius)
            kernels = kernel if kernels is None else kernel + kernels
        return kernels

    @staticmethod
    def _create_kernels_from_features(features, radius):
        assert features.dim() == 4, 'Features must be a NCHW batch'
        N, C, H, W = features.shape
        kernels = LocalSaliencyCoherence._unfold(features, radius)
        kernels = kernels - kernels[:, :, radius, radius, :, :].view(N, C, 1, 1, H, W)
        kernels = (-0.5 * kernels ** 2).sum(dim=1, keepdim=True).exp()
        return kernels

    @staticmethod
    def _get_mesh(N, H, W, device):
        return torch.cat((
            torch.arange(0, W, 1, dtype=torch.float32, device=device).view(1, 1, 1, W).repeat(N, 1, H, 1),
            torch.arange(0, H, 1, dtype=torch.float32, device=device).view(1, 1, H, 1).repeat(N, 1, 1, W)
        ), 1)

    @staticmethod
    def _unfold(img, radius):
        assert img.dim() == 4, 'Unfolding requires NCHW batch'
        N, C, H, W = img.shape
        diameter = 2 * radius + 1
        return F.unfold(img, diameter, 1, radius).view(N, C, diameter, diameter, H, W)

    @staticmethod
    def _visualize_kernels(kernels, radius, height_input, width_input, height_pred, width_pred):
        diameter = 2 * radius + 1
        vis = kernels[:, :, :, :, radius::diameter, radius::diameter]
        vis_nh, vis_nw = vis.shape[-2:]
        vis = vis.permute(0, 1, 4, 2, 5, 3).contiguous().view(kernels.shape[0], 1, diameter * vis_nh, diameter * vis_nw)
        if vis.shape[2] > height_pred:
            vis = vis[:, :, :height_pred, :]
        if vis.shape[3] > width_pred:
            vis = vis[:, :, :, :width_pred]
        if vis.shape[2:] != (height_pred, width_pred):
            vis = F.pad(vis, [0, width_pred-vis.shape[3], 0, height_pred-vis.shape[2]])
        vis = F.interpolate(vis, (height_input, width_input), mode='nearest')
        return vis


# [DISABLED:masked_uncertainty_edge_loss] def masked_uncertainty_edge_loss(pred_edge_logits, canny_gt, pred_mask_prob):
# [DISABLED:masked_uncertainty_edge_loss]     """
# [DISABLED:masked_uncertainty_edge_loss]     边缘损失 - 直接计算，无warmup
# [DISABLED:masked_uncertainty_edge_loss]     """
# [DISABLED:masked_uncertainty_edge_loss]     # 1. 保持 kernel_size=9，给 Canny 边缘留出充足的容错空间
# [DISABLED:masked_uncertainty_edge_loss]     dilated = F.max_pool2d(pred_mask_prob, kernel_size=9, stride=1, padding=4)
# [DISABLED:masked_uncertainty_edge_loss]     eroded = -F.max_pool2d(-pred_mask_prob, kernel_size=9, stride=1, padding=4)
# [DISABLED:masked_uncertainty_edge_loss]     boundary_band = dilated - eroded
# [DISABLED:masked_uncertainty_edge_loss]     boundary_mask = (boundary_band > 0.1).float().detach()
#
# [DISABLED:masked_uncertainty_edge_loss]     # 2. 尺寸对齐
# [DISABLED:masked_uncertainty_edge_loss]     if pred_edge_logits.shape[-2:] != canny_gt.shape[-2:]:
# [DISABLED:masked_uncertainty_edge_loss]         canny_gt = F.interpolate(canny_gt, size=pred_edge_logits.shape[-2:], mode='nearest')
# [DISABLED:masked_uncertainty_edge_loss]     if boundary_mask.shape[-2:] != pred_edge_logits.shape[-2:]:
# [DISABLED:masked_uncertainty_edge_loss]         boundary_mask = F.interpolate(boundary_mask, size=pred_edge_logits.shape[-2:], mode='bilinear')
#
# [DISABLED:masked_uncertainty_edge_loss]     # 3. 清洗 GT
# [DISABLED:masked_uncertainty_edge_loss]     clean_canny = (canny_gt * boundary_mask).detach()
#
# [DISABLED:masked_uncertainty_edge_loss]     # 4. 全局 BCE Loss
# [DISABLED:masked_uncertainty_edge_loss]     pos_weight = torch.tensor([15.0], device=pred_edge_logits.device)
# [DISABLED:masked_uncertainty_edge_loss]     bce_loss = F.binary_cross_entropy_with_logits(
# [DISABLED:masked_uncertainty_edge_loss]         pred_edge_logits, clean_canny, pos_weight=pos_weight, reduction='mean'
# [DISABLED:masked_uncertainty_edge_loss]     )
#
# [DISABLED:masked_uncertainty_edge_loss]     # 5. Dice Loss 辅助收敛
# [DISABLED:masked_uncertainty_edge_loss]     pred_prob = torch.sigmoid(pred_edge_logits)
# [DISABLED:masked_uncertainty_edge_loss]     intersection = (pred_prob * clean_canny).sum(dim=(2, 3))
# [DISABLED:masked_uncertainty_edge_loss]     union = pred_prob.sum(dim=(2, 3)) + clean_canny.sum(dim=(2, 3))
# [DISABLED:masked_uncertainty_edge_loss]     dice_loss = 1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)
#
# [DISABLED:masked_uncertainty_edge_loss]     final_loss = bce_loss + dice_loss.mean()
#
# [DISABLED:masked_uncertainty_edge_loss]     return final_loss, boundary_mask
#

def compute_project_term(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    inter = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    return 1.0 - (2.0 * inter / (union + 1e-6))


def projection_loss(pred_logits: torch.Tensor, boxes: torch.Tensor, *, tau: float = 10.0) -> torch.Tensor:
    """强制预测掩码在 x 和 y 轴上的投影与 BBox 一致，支持单框或多框。"""
    B, _, H, W = pred_logits.shape
    device = pred_logits.device

    if boxes.dim() == 1:
        boxes = boxes.view(1, 1, 4)
    elif boxes.dim() == 2:
        if boxes.shape[-1] != 4:
            raise ValueError(f"boxes must end with 4 coords, got {tuple(boxes.shape)}")
        boxes = boxes.unsqueeze(0)
    elif boxes.dim() != 3 or boxes.shape[-1] != 4:
        raise ValueError(f"Unsupported boxes shape: {tuple(boxes.shape)}")

    if boxes.shape[0] == 1 and B > 1:
        boxes = boxes.expand(B, -1, -1)
    elif boxes.shape[0] != B:
        raise ValueError(f"Batch size mismatch: pred batch {B}, boxes batch {boxes.shape[0]}")

    pred_x = torch.logsumexp(pred_logits * tau, dim=2).squeeze(1) / tau
    pred_y = torch.logsumexp(pred_logits * tau, dim=3).squeeze(1) / tau

    x_coords = torch.arange(W, device=device).view(1, 1, W)
    y_coords = torch.arange(H, device=device).view(1, 1, H)

    x1, y1, x2, y2 = boxes[..., 0:1], boxes[..., 1:2], boxes[..., 2:3], boxes[..., 3:4]

    mask_x = ((x_coords >= x1) & (x_coords <= x2)).float().max(dim=1).values
    mask_y = ((y_coords >= y1) & (y_coords <= y2)).float().max(dim=1).values

    return F.binary_cross_entropy_with_logits(pred_x, mask_x) + F.binary_cross_entropy_with_logits(pred_y, mask_y)


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred_prob = torch.sigmoid(pred)
    inter = ((pred_prob * mask) * weit).sum(dim=(2, 3))
    union = ((pred_prob + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def uncertainty_distillation_loss(pred_logits, sam_logits, gt_mask):
    if sam_logits.shape[-2:] != pred_logits.shape[-2:]:
        sam_logits = F.interpolate(sam_logits, size=pred_logits.shape[-2:], mode='bilinear', align_corners=False)

    with torch.no_grad():
        sam_prob = torch.sigmoid(sam_logits)

        # 越接近 0 或 1，权重越大；越接近 0.5，权重越小
        uncertainty_weight = torch.abs(2 * sam_prob - 1).pow(2.0)

        valid_region = 1.0 - gt_mask

        final_weight = uncertainty_weight * valid_region

    loss_map = F.binary_cross_entropy_with_logits(pred_logits, sam_prob, reduction='none')

    loss = (loss_map * final_weight).sum() / (final_weight.sum() + 1e-8)

    return loss


class LayerNorm2d(nn.Module):
    """2D LayerNorm"""
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MLP(nn.Module):
    """多层感知机"""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x




# [DISABLED:UncertaintyAwareModule] class UncertaintyAwareModule(nn.Module):
# [DISABLED:UncertaintyAwareModule]     """不确定性感知模块 - 基于原型学习和LDA思想
# [DISABLED:UncertaintyAwareModule]     输入：features(B, 32, 288, 288) + logits(B, 1, 288, 288) + pseudo_labels(B, 288, 288)
# [DISABLED:UncertaintyAwareModule]     输出：weighted_loss + margin_loss + uncertainty_map
# [DISABLED:UncertaintyAwareModule]     """
# [DISABLED:UncertaintyAwareModule]     def __init__(self, num_classes=2, feature_dim=32, momentum=0.99, temperature=0.1):
# [DISABLED:UncertaintyAwareModule]         super().__init__()
# [DISABLED:UncertaintyAwareModule]         self.num_classes = num_classes
# [DISABLED:UncertaintyAwareModule]         self.feature_dim = feature_dim
# [DISABLED:UncertaintyAwareModule]         self.momentum = momentum
# [DISABLED:UncertaintyAwareModule]         self.temperature = temperature
# [DISABLED:UncertaintyAwareModule]         self.register_buffer('prototypes', torch.zeros(num_classes, feature_dim))
# [DISABLED:UncertaintyAwareModule]         self.initialized = False
#
# [DISABLED:UncertaintyAwareModule]     def update_prototypes(self, features, pseudo_labels):
# [DISABLED:UncertaintyAwareModule]         """EMA更新原型"""
# [DISABLED:UncertaintyAwareModule]         with torch.no_grad():
# [DISABLED:UncertaintyAwareModule]             B, C, H, W = features.shape
# [DISABLED:UncertaintyAwareModule]             features_flat = features.permute(0, 2, 3, 1).reshape(-1, C)
# [DISABLED:UncertaintyAwareModule]             labels_flat = pseudo_labels.view(-1)
#
# [DISABLED:UncertaintyAwareModule]             # 创建有效掩码
# [DISABLED:UncertaintyAwareModule]             valid_mask = (labels_flat >= 0) & (labels_flat < self.num_classes)
# [DISABLED:UncertaintyAwareModule]             features_flat = features_flat[valid_mask]
# [DISABLED:UncertaintyAwareModule]             labels_flat = labels_flat[valid_mask]
#
# [DISABLED:UncertaintyAwareModule]             # 初始化原型
# [DISABLED:UncertaintyAwareModule]             if not self.initialized:
# [DISABLED:UncertaintyAwareModule]                 for cls_idx in range(self.num_classes):
# [DISABLED:UncertaintyAwareModule]                     cls_mask = (labels_flat == cls_idx)
# [DISABLED:UncertaintyAwareModule]                     if cls_mask.sum() > 0:
# [DISABLED:UncertaintyAwareModule]                         self.prototypes[cls_idx] = features_flat[cls_mask].mean(dim=0)
# [DISABLED:UncertaintyAwareModule]                 self.initialized = True
# [DISABLED:UncertaintyAwareModule]                 return
#
# [DISABLED:UncertaintyAwareModule]             # EMA更新
# [DISABLED:UncertaintyAwareModule]             for cls_idx in range(self.num_classes):
# [DISABLED:UncertaintyAwareModule]                 cls_mask = (labels_flat == cls_idx)
# [DISABLED:UncertaintyAwareModule]                 if cls_mask.sum() > 0:
# [DISABLED:UncertaintyAwareModule]                     cls_features = features_flat[cls_mask]
# [DISABLED:UncertaintyAwareModule]                     cls_mean = cls_features.mean(dim=0)
# [DISABLED:UncertaintyAwareModule]                     self.prototypes[cls_idx] = (
# [DISABLED:UncertaintyAwareModule]                             self.momentum * self.prototypes[cls_idx] +
# [DISABLED:UncertaintyAwareModule]                             (1 - self.momentum) * cls_mean
# [DISABLED:UncertaintyAwareModule]                     )
#
# [DISABLED:UncertaintyAwareModule]     def compute_uncertainty(self, features, pseudo_labels):
# [DISABLED:UncertaintyAwareModule]         """计算不确定性"""
# [DISABLED:UncertaintyAwareModule]         B, C, H, W = features.shape
# [DISABLED:UncertaintyAwareModule]         features_flat = features.permute(0, 2, 3, 1).reshape(-1, C)
# [DISABLED:UncertaintyAwareModule]         labels_flat = pseudo_labels.view(-1)
#
# [DISABLED:UncertaintyAwareModule]         # 创建有效掩码
# [DISABLED:UncertaintyAwareModule]         valid_mask = (labels_flat >= 0) & (labels_flat < self.num_classes)
# [DISABLED:UncertaintyAwareModule]         features_valid = features_flat[valid_mask]
# [DISABLED:UncertaintyAwareModule]         labels_valid = labels_flat[valid_mask]
#
# [DISABLED:UncertaintyAwareModule]         # 初始化不确定性张量（使用与features相同的dtype）
# [DISABLED:UncertaintyAwareModule]         intra_sim_full = torch.zeros(B * H * W, device=features.device, dtype=features.dtype)
# [DISABLED:UncertaintyAwareModule]         inter_sim_full = torch.zeros(B * H * W, device=features.device, dtype=features.dtype)
#
# [DISABLED:UncertaintyAwareModule]         if len(features_valid) > 0:
# [DISABLED:UncertaintyAwareModule]             # 归一化特征和原型
# [DISABLED:UncertaintyAwareModule]             norm_features = F.normalize(features_valid, p=2, dim=1)
# [DISABLED:UncertaintyAwareModule]             norm_prototypes = F.normalize(self.prototypes, p=2, dim=1)
#
# [DISABLED:UncertaintyAwareModule]             # 计算所有相似度
# [DISABLED:UncertaintyAwareModule]             all_sim = norm_features @ norm_prototypes.T  # [N_valid, num_classes]
#
# [DISABLED:UncertaintyAwareModule]             # 类内相似度
# [DISABLED:UncertaintyAwareModule]             intra_sim = all_sim[torch.arange(len(labels_valid)), labels_valid]
#
# [DISABLED:UncertaintyAwareModule]             # 类间相似度（其他类的最大相似度）
# [DISABLED:UncertaintyAwareModule]             mask = torch.ones_like(all_sim, dtype=torch.bool)
# [DISABLED:UncertaintyAwareModule]             mask[torch.arange(len(labels_valid)), labels_valid] = False
# [DISABLED:UncertaintyAwareModule]             inter_sim = all_sim[mask].view(len(labels_valid), -1).max(dim=1)[0]
#
# [DISABLED:UncertaintyAwareModule]             # 填充有效位置
# [DISABLED:UncertaintyAwareModule]             intra_sim_full[valid_mask] = intra_sim
# [DISABLED:UncertaintyAwareModule]             inter_sim_full[valid_mask] = inter_sim
#
# [DISABLED:UncertaintyAwareModule]         # 重塑为空间维度
# [DISABLED:UncertaintyAwareModule]         intra_sim_full = intra_sim_full.view(B, H, W)
# [DISABLED:UncertaintyAwareModule]         inter_sim_full = inter_sim_full.view(B, H, W)
#
# [DISABLED:UncertaintyAwareModule]         # 计算margin和不确定性
# [DISABLED:UncertaintyAwareModule]         sim_margin = intra_sim_full - inter_sim_full
# [DISABLED:UncertaintyAwareModule]         uncertainty = 1 - torch.sigmoid(sim_margin / self.temperature)
#
# [DISABLED:UncertaintyAwareModule]         return uncertainty, intra_sim_full, inter_sim_full
#
# [DISABLED:UncertaintyAwareModule]     def forward(self, features, logits, pseudo_labels):
# [DISABLED:UncertaintyAwareModule]         """
# [DISABLED:UncertaintyAwareModule]         Args:
# [DISABLED:UncertaintyAwareModule]             features: (B, 32, 288, 288) - 浅层特征
# [DISABLED:UncertaintyAwareModule]             logits: (B, 1, 288, 288) - 预测logits
# [DISABLED:UncertaintyAwareModule]             pseudo_labels: (B, 288, 288) - 伪标签(0/1)
# [DISABLED:UncertaintyAwareModule]         Returns:
# [DISABLED:UncertaintyAwareModule]             weighted_loss: 加权CE损失
# [DISABLED:UncertaintyAwareModule]             margin_loss: 原型对齐损失
# [DISABLED:UncertaintyAwareModule]             uncertainty: (B, 1, 288, 288) - 不确定性图
# [DISABLED:UncertaintyAwareModule]         """
# [DISABLED:UncertaintyAwareModule]         # 更新原型
# [DISABLED:UncertaintyAwareModule]         self.update_prototypes(features.detach(), pseudo_labels)
#
# [DISABLED:UncertaintyAwareModule]         # 计算不确定性
# [DISABLED:UncertaintyAwareModule]         uncertainty, _, _ = self.compute_uncertainty(features.detach(), pseudo_labels)
#
# [DISABLED:UncertaintyAwareModule]         # 加权CE损失
# [DISABLED:UncertaintyAwareModule]         ce_loss = F.binary_cross_entropy_with_logits(logits, pseudo_labels.float().unsqueeze(1), reduction='none')
# [DISABLED:UncertaintyAwareModule]         weighted_loss = ((1 - uncertainty.unsqueeze(1)) * ce_loss).mean()
#
# [DISABLED:UncertaintyAwareModule]         # 原型对齐损失（margin增强）
# [DISABLED:UncertaintyAwareModule]         B, C, H, W = features.shape
# [DISABLED:UncertaintyAwareModule]         features_flat = features.permute(0, 2, 3, 1).reshape(-1, C)
# [DISABLED:UncertaintyAwareModule]         labels_flat = pseudo_labels.view(-1)
# [DISABLED:UncertaintyAwareModule]         valid_mask = (labels_flat >= 0) & (labels_flat < self.num_classes)
#
# [DISABLED:UncertaintyAwareModule]         if valid_mask.any():
# [DISABLED:UncertaintyAwareModule]             norm_features = F.normalize(features_flat[valid_mask], p=2, dim=1)
# [DISABLED:UncertaintyAwareModule]             norm_prototypes = F.normalize(self.prototypes, p=2, dim=1)
# [DISABLED:UncertaintyAwareModule]             target_prototypes = norm_prototypes[labels_flat[valid_mask]]
# [DISABLED:UncertaintyAwareModule]             target_sim = (norm_features * target_prototypes).sum(dim=1)
# [DISABLED:UncertaintyAwareModule]             margin_loss = F.relu(0.5 - target_sim).mean()
# [DISABLED:UncertaintyAwareModule]         else:
# [DISABLED:UncertaintyAwareModule]             margin_loss = torch.tensor(0.0, device=features.device)
#
# [DISABLED:UncertaintyAwareModule]         # 返回uncertainty_map时添加通道维度 (B, 288, 288) -> (B, 1, 288, 288)
# [DISABLED:UncertaintyAwareModule]         return weighted_loss, margin_loss, uncertainty.unsqueeze(1)
#

# [DISABLED:HQFusionAdapter] class HQFusionAdapter(nn.Module):
# [DISABLED:HQFusionAdapter]     """HQ Fusion Adapter - Edge-boosted mask refinement (Scheme C)
#
# [DISABLED:HQFusionAdapter]     Core idea:
# [DISABLED:HQFusionAdapter]       SAM mask logits are uncertain (~0) at object boundaries.
# [DISABLED:HQFusionAdapter]       GEA edge_prob is high (~1) exactly at those boundaries.
# [DISABLED:HQFusionAdapter]       We boost the logits at boundary pixels in the correct direction:
# [DISABLED:HQFusionAdapter]         - fg-side edge (tanh(logit)>0): push logit higher  -> sigmoid->1
# [DISABLED:HQFusionAdapter]         - bg-side edge (tanh(logit)<0): push logit lower   -> sigmoid->0
#
# [DISABLED:HQFusionAdapter]     Formula:
# [DISABLED:HQFusionAdapter]       soft_dir  = tanh(sam_mask)          # smooth sign: +1 fg, -1 bg, 0 at boundary
# [DISABLED:HQFusionAdapter]       boost     = edge_prob * soft_dir * k  # edge_prob gates spatial location
# [DISABLED:HQFusionAdapter]       hq_mask   = sam_mask + boost
#
# [DISABLED:HQFusionAdapter]     tanh vs sign:
# [DISABLED:HQFusionAdapter]       tanh avoids discontinuity at logit=0 (boundary pixels)
# [DISABLED:HQFusionAdapter]       At large |logit|: tanh≈sign (same behavior in confident regions)
# [DISABLED:HQFusionAdapter]       At logit≈0:       tanh≈logit (smooth gradient, no artifacts)
#
# [DISABLED:HQFusionAdapter]     Input:  sam_mask_low(B,1,288,288) + edge_logits(B,1,288,288)
# [DISABLED:HQFusionAdapter]             refined_shallow is accepted but not used (kept for API compat)
# [DISABLED:HQFusionAdapter]     Output: hq_mask_high(B,1,288,288)
# [DISABLED:HQFusionAdapter]     """
# [DISABLED:HQFusionAdapter]     def __init__(self, feat_dim=32, transformer_dim=256):
# [DISABLED:HQFusionAdapter]         super().__init__()
# [DISABLED:HQFusionAdapter]         # Learnable edge boost strength (init 2.0: moderate boost)
# [DISABLED:HQFusionAdapter]         self.edge_boost_k = nn.Parameter(torch.tensor(2.0))
#
# [DISABLED:HQFusionAdapter]     def forward(self, sam_mask_low, refined_shallow, edge_logits):
# [DISABLED:HQFusionAdapter]         """
# [DISABLED:HQFusionAdapter]         Args:
# [DISABLED:HQFusionAdapter]             sam_mask_low:    (B, 1, 288, 288) - SAM decoder output logits
# [DISABLED:HQFusionAdapter]             refined_shallow: (B, 32, 288, 288) - not used (API compat)
# [DISABLED:HQFusionAdapter]             edge_logits:     (B, 1, 288, 288) - GEA edge logits
# [DISABLED:HQFusionAdapter]         Returns:
# [DISABLED:HQFusionAdapter]             hq_mask_high: (B, 1, 288, 288) - edge-boosted mask logits
# [DISABLED:HQFusionAdapter]         """
# [DISABLED:HQFusionAdapter]         # Edge probability: high at boundaries, low elsewhere
# [DISABLED:HQFusionAdapter]         edge_prob = torch.sigmoid(edge_logits)          # (B, 1, 288, 288), [0, 1]
#
# [DISABLED:HQFusionAdapter]         # Soft direction: +1 for fg-side, -1 for bg-side, smooth at boundary
# [DISABLED:HQFusionAdapter]         soft_dir = torch.tanh(sam_mask_low)             # (B, 1, 288, 288), [-1, 1]
#
# [DISABLED:HQFusionAdapter]         # Boost: at edge pixels, push logits in the correct direction
# [DISABLED:HQFusionAdapter]         boost = edge_prob * soft_dir * self.edge_boost_k  # (B, 1, 288, 288)
#
# [DISABLED:HQFusionAdapter]         hq_mask_high = sam_mask_low + boost
#
# [DISABLED:HQFusionAdapter]         return hq_mask_high
#

# [DISABLED:hq_structure_loss] def hq_structure_loss(pred_logits: torch.Tensor, target_prob: torch.Tensor) -> torch.Tensor:
# [DISABLED:hq_structure_loss]     """HQ结构损失 - 让HQ mask学习prompt mask的结构"""
# [DISABLED:hq_structure_loss]     return structure_loss(pred_logits, target_prob.detach())
#

# [DISABLED:hq_edge_aware_loss] def hq_edge_aware_loss(hq_logits: torch.Tensor, edge_logits: torch.Tensor, target_prob: torch.Tensor) -> torch.Tensor:
# [DISABLED:hq_edge_aware_loss]     """HQ边缘感知损失 - 在边缘区域加强监督"""
# [DISABLED:hq_edge_aware_loss]     # 对齐分辨率
# [DISABLED:hq_edge_aware_loss]     target_size = hq_logits.shape[-2:]
# [DISABLED:hq_edge_aware_loss]     edge_logits_aligned = F.interpolate(edge_logits, size=target_size, mode='bilinear', align_corners=False)
#
# [DISABLED:hq_edge_aware_loss]     edge_prob = torch.sigmoid(edge_logits_aligned)
#
# [DISABLED:hq_edge_aware_loss]     # 在边缘区域加大权重
# [DISABLED:hq_edge_aware_loss]     edge_weight = 1.0 + 2.0 * edge_prob  # 背景权重1.0，边缘权重3.0
#
# [DISABLED:hq_edge_aware_loss]     # 计算加权BCE损失
# [DISABLED:hq_edge_aware_loss]     bce_loss = F.binary_cross_entropy_with_logits(hq_logits, target_prob.detach(), reduction='none')
# [DISABLED:hq_edge_aware_loss]     weighted_loss = (bce_loss * edge_weight).mean()
#
# [DISABLED:hq_edge_aware_loss]     return weighted_loss
#


class UncertaintyAwareModule(nn.Module):
    """Uncertainty-aware prototype module for binary (fg/bg) segmentation.

    Input features: noprompt decoder output features (B, C, H, W).
    Prototypes updated via EMA from prompt pseudo-labels.
    Uncertainty = 1 - sigmoid((intra_sim - inter_sim) / temperature)
    """
    def __init__(self, feature_dim=32, momentum=0.995, temperature=0.1):
        super().__init__()
        self.num_classes  = 2
        self.feature_dim  = feature_dim
        self.momentum     = momentum
        self.temperature  = temperature
        self.register_buffer('prototypes', torch.zeros(2, feature_dim))
        self.initialized  = False

    def update_prototypes(self, features, pseudo_labels):
        """EMA update of fg/bg prototypes from labelled pixels."""
        with torch.no_grad():
            B, C, H, W = features.shape
            feat_flat   = features.permute(0, 2, 3, 1).reshape(-1, C).to(torch.float32)
            label_flat  = pseudo_labels.view(-1)
            valid_mask  = (label_flat >= 0) & (label_flat < self.num_classes)
            feat_valid  = feat_flat[valid_mask]
            label_valid = label_flat[valid_mask]
            if not self.initialized:
                for cls in range(self.num_classes):
                    m = (label_valid == cls)
                    if m.sum() > 0:
                        self.prototypes[cls] = feat_valid[m].mean(dim=0)
                if (label_valid >= 0).sum() > 0:
                    self.initialized = True
                return
            for cls in range(self.num_classes):
                m = (label_valid == cls)
                if m.sum() > 0:
                    self.prototypes[cls] = (
                        self.momentum * self.prototypes[cls]
                        + (1 - self.momentum) * feat_valid[m].mean(dim=0)
                    )

    def compute_uncertainty(self, features, pseudo_labels):
        """Per-pixel uncertainty from intra/inter prototype similarity margin."""
        B, C, H, W = features.shape
        feat_flat   = features.permute(0, 2, 3, 1).reshape(-1, C).to(torch.float32)
        label_flat  = pseudo_labels.view(-1)
        valid_mask  = (label_flat >= 0) & (label_flat < self.num_classes)
        intra_full  = torch.zeros(B * H * W, device=features.device, dtype=torch.float32)
        inter_full  = torch.zeros(B * H * W, device=features.device, dtype=torch.float32)
        if valid_mask.sum() > 0 and self.initialized:
            feat_v      = F.normalize(feat_flat[valid_mask], p=2, dim=1)
            proto       = F.normalize(self.prototypes.to(torch.float32), p=2, dim=1)
            all_sim     = feat_v @ proto.T                          # (N_valid, 2)
            label_valid = label_flat[valid_mask]
            intra = all_sim[torch.arange(len(label_valid)), label_valid]
            other = torch.ones_like(all_sim, dtype=torch.bool)
            other[torch.arange(len(label_valid)), label_valid] = False
            inter = all_sim[other].view(len(label_valid), -1).max(dim=1)[0]
            intra_full[valid_mask] = intra.to(torch.float32)
            inter_full[valid_mask] = inter.to(torch.float32)
        margin      = (intra_full - inter_full).view(B, H, W)
        uncertainty = 1.0 - torch.sigmoid(margin / self.temperature)
        return uncertainty

    def compute_margin_loss(self, features, pseudo_labels):
        """Prototype alignment loss: push features toward their class prototype."""
        B, C, H, W = features.shape
        feat_flat   = features.permute(0, 2, 3, 1).reshape(-1, C)
        label_flat  = pseudo_labels.view(-1)
        valid_mask  = (label_flat >= 0) & (label_flat < self.num_classes)
        if not self.initialized or valid_mask.sum() == 0:
            return torch.tensor(0.0, device=features.device)
        feat_v       = F.normalize(feat_flat[valid_mask].to(torch.float32), p=2, dim=1)
        proto        = F.normalize(self.prototypes.to(torch.float32), p=2, dim=1)
        target_proto = proto[label_flat[valid_mask]]           # (N_valid, C)
        target_sim   = (feat_v * target_proto).sum(dim=1).to(torch.float32)      # (N_valid,)
        return F.relu(0.5 - target_sim).mean()

    def forward(self, features, pseudo_labels):
        """Update prototypes, compute uncertainty and margin loss.
        Args:
            features:      (B, C, H, W)  noprompt decoder features
            pseudo_labels: (B, H, W)     0=bg, 1=fg, -1=ignore
        Returns:
            uncertainty:  (B, H, W)  in [0, 1]
            margin_loss:  scalar tensor
        """
        self.update_prototypes(features.detach(), pseudo_labels)
        uncertainty  = self.compute_uncertainty(features.detach(), pseudo_labels)
        margin_loss  = self.compute_margin_loss(features, pseudo_labels)  # has gradient
        return uncertainty, margin_loss


class PromptLoss(nn.Module):
    def __init__(self, opt: argparse.Namespace):
        super().__init__()
        self.opt = opt

        self.smooth_loss_fn = SmoothnessLoss()
        self.lsc_loss_fn = LocalSaliencyCoherence()

        self.lsc_kernels = [{"weight": 1, "xy": 6, "rgb": 0.1}]
        self.lsc_radius = 5
        # 使用原型不确定性感知模块，替代 masked_uncertainty_edge_loss 对 prompt 分支的监督
        # [DISABLED] self.uncertainty_module = UncertaintyAwareModule(num_classes=2, feature_dim=32)

        self._loss_dispatch = {
            'scribble': self._calc_loss_scribble,
            'box': self._calc_loss_box,
            'point': self._calc_loss_point
        }

    def forward(self, prompt_type: str, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        handler = self._loss_dispatch.get(prompt_type)
        if handler is None:
            raise ValueError(f"Unsupported prompt type: {prompt_type}")

        return handler(**kwargs)

    def _calc_loss_scribble(
            self,
            pred_logits: torch.Tensor,
            pred_prob: torch.Tensor,
            gt_mask: torch.Tensor,
            mask_weight: torch.Tensor,
            gray_image: torch.Tensor,
            ratio: float,
            rgb_input: Optional[torch.Tensor],
            pred_multi_prob: Optional[torch.Tensor] = None,
            pred_edge_logits: Optional[torch.Tensor] = None,
            canny_edge_gt: Optional[torch.Tensor] = None,
            refined_shallow_feats: Optional[torch.Tensor] = None,
            hq_mask_logits: Optional[torch.Tensor] = None,
            **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        loss_dict = {}

        # 1. Saliency Loss
        bce_map = F.binary_cross_entropy_with_logits(pred_logits, gt_mask, reduction="none")
        sal_loss = ratio * (bce_map * mask_weight).sum() / (mask_weight.sum() + 1e-8)
        loss_dict['sal'] = sal_loss.item()

        # 2. Smoothness Loss
        sm_loss = self.opt.sm_loss_weight * self.smooth_loss_fn(pred_prob, gray_image)
        loss_dict['sm'] = sm_loss.item()

        # 3. LSC Loss
        target_h, target_w = gt_mask.shape[2] // 4, gt_mask.shape[3] // 4
        sample_rgb = {"rgb": F.interpolate(rgb_input, size=(target_h, target_w), mode="bilinear", align_corners=False)}
        prob_low = F.interpolate(pred_prob, size=(target_h, target_w), mode="bilinear", align_corners=False)
        lsc_res = self.lsc_loss_fn(prob_low, self.lsc_kernels, self.lsc_radius, sample_rgb, target_h, target_w)
        lsc_loss = lsc_res["loss"]
        loss_dict['lsc'] = lsc_loss.item()

        # 4. Edge Loss（边缘监督）
        # [DISABLED] edge_loss = torch.tensor(0.0, device=pred_logits.device)
        # [DISABLED] if pred_edge_logits is not None and canny_edge_gt is not None:
        # [DISABLED] edge_loss, _ = masked_uncertainty_edge_loss(
        # [DISABLED] pred_edge_logits, canny_edge_gt, pred_prob
        # [DISABLED] )
        # [DISABLED] loss_dict['edge'] = edge_loss.item()
        # [DISABLED] 
        # 5. Uncertainty-Aware Loss [DISABLED]
        unc_aware_loss = torch.tensor(0.0, device=pred_logits.device)

        # 6. HQ Loss [DISABLED]
        hq_loss = torch.tensor(0.0, device=pred_logits.device)


        total_loss = sal_loss + sm_loss + lsc_loss  # [edge/unc/hq DISABLED]
        return total_loss, loss_dict

    def _calc_loss_box(
            self,
            pred_prob: torch.Tensor,
            gt_mask: torch.Tensor,
            gray_image: torch.Tensor,
            rgb_input: torch.Tensor,
            input_boxes: Optional[torch.Tensor] = None,
            **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_dict = {}

        # 1. Projection Loss
        proj_loss = projection_loss(pred_prob, input_boxes)
        loss_dict['box_proj'] = proj_loss.item()

        # 2. Smoothness Loss
        sm_loss = self.opt.sm_loss_weight * self.smooth_loss_fn(pred_prob, gray_image)
        loss_dict['box_sm'] = sm_loss.item()

        # 3. LSC Loss
        target_h, target_w = gt_mask.shape[2] // 4, gt_mask.shape[3] // 4

        sample_rgb = {"rgb": F.interpolate(rgb_input, size=(target_h, target_w), mode="bilinear", align_corners=False)}
        prob_low = F.interpolate(pred_prob, size=(target_h, target_w), mode="bilinear", align_corners=False)

        lsc_res = self.lsc_loss_fn(prob_low, self.lsc_kernels, self.lsc_radius, sample_rgb, target_h, target_w)
        lsc_loss = lsc_res["loss"]
        loss_dict['box_lsc'] = lsc_loss.item()

        total_loss = proj_loss + sm_loss + lsc_loss

        return total_loss, loss_dict

    def _calc_loss_point(self, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        pass


class NopromptLoss(nn.Module):
    def __init__(self, opt: argparse.Namespace):
        super().__init__()
        self.opt = opt
        self.unc_module = UncertaintyAwareModule(feature_dim=32, momentum=0.995, temperature=0.1)

        self._loss_dispatch = {
            'scribble': self._calc_loss_scribble,
            'box': self._calc_loss_box,
            'point': self._calc_loss_point
        }

    def forward(
            self,
            prompt_type: str,
            pred_noprompt_logits: torch.Tensor,
            pred_prompt_prob: torch.Tensor,
            current_unc_weight: Optional[float] = None,
            **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        handler = self._loss_dispatch.get(prompt_type)
        if handler is None:
            raise ValueError(f"Unsupported prompt type: {prompt_type}")

        return handler(
            pred_noprompt_logits=pred_noprompt_logits,
            pred_prompt_prob=pred_prompt_prob,
            uncertainty_weight=current_unc_weight,
            **kwargs
        )

    def _calc_loss_scribble(
            self,
            pred_noprompt_logits: torch.Tensor,
            pred_prompt_prob: torch.Tensor,
            uncertainty_weight: float,
            mask_weight: torch.Tensor,
            unc_logits: torch.Tensor,
            refined_shallow_feats: Optional[torch.Tensor] = None,
            pred_edge_logits: Optional[torch.Tensor] = None,
            canny_edge_gt: Optional[torch.Tensor] = None,
            noprompt_hq_mask_logits: Optional[torch.Tensor] = None,
            gt_mask: Optional[torch.Tensor] = None,
            ratio: float = 1.0,
            high_res_feats: Optional[torch.Tensor] = None,
            noprompt_decoder_feats: Optional[torch.Tensor] = None,
            **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:


        loss_dict = {}
        pred_noprompt_prob = torch.sigmoid(pred_noprompt_logits)

        # ── 1. Structure Loss (align noprompt with prompt structure) ──────────
        str_loss = structure_loss(pred_noprompt_logits, pred_prompt_prob.detach())
        loss_dict['noprompt_str'] = str_loss.item()

        # ── 2. Uncertainty-Aware Loss ─────────────────────────────────────────
        #
        # Uses noprompt decoder output features (32-dim, 288x288).
        # Pseudo-labels from prompt_prob to update fg/bg prototypes (EMA).
        # Uncertainty = 1 - sigmoid((intra_sim - inter_sim) / T)
        # Weighted BCE: certain regions (low uncertainty) get high weight.
        unc_loss    = torch.tensor(0.0, device=pred_noprompt_logits.device)
        margin_loss = torch.tensor(0.0, device=pred_noprompt_logits.device)
        uncertainty = None
        pseudo_labels = None

        if noprompt_decoder_feats is not None:
            feat = noprompt_decoder_feats  # (B, 32, H, W) - no detach, margin_loss needs gradient
            feat_size = pred_noprompt_logits.shape[-2:]
            if feat.shape[-2:] != feat_size:
                feat = F.interpolate(feat, size=feat_size, mode='bilinear', align_corners=False)

            # Build pseudo-labels from prompt_prob
            prompt_prob_aligned = F.interpolate(pred_prompt_prob, size=feat_size,
                                                mode='bilinear', align_corners=False).squeeze(1)
            pseudo_labels = torch.full_like(prompt_prob_aligned, -1, dtype=torch.long)
            pseudo_labels[prompt_prob_aligned > 0.6] = 1   # confident fg
            pseudo_labels[prompt_prob_aligned < 0.4] = 0   # confident bg

            # Update prototypes and compute uncertainty + margin_loss
            uncertainty, margin_loss = self.unc_module(feat, pseudo_labels)

            if self.unc_module.initialized:
                # Uncertainty-weighted BCE against prompt_target
                certainty = (1.0 - uncertainty).unsqueeze(1).detach()  # (B,1,H,W)
                prompt_target = F.interpolate(pred_prompt_prob, size=feat_size,
                                              mode='bilinear', align_corners=False).detach()
                noprompt_aligned = F.interpolate(pred_noprompt_logits, size=feat_size,
                                                 mode='bilinear', align_corners=False)
                bce_raw  = F.binary_cross_entropy_with_logits(
                    noprompt_aligned, prompt_target, reduction='none'
                )
                unc_loss = (certainty * bce_raw).sum() / (certainty.sum() + 1e-8)

        loss_dict['noprompt_unc']    = unc_loss.item()
        loss_dict['noprompt_margin'] = margin_loss.item()

        # Visualizations
        if noprompt_decoder_feats is not None and self.unc_module.initialized and uncertainty is not None:
            loss_dict['vis_uncertainty']   = uncertainty.unsqueeze(1).detach()
            loss_dict['vis_pseudo_labels'] = ((pseudo_labels.float() + 1) / 2).unsqueeze(1).detach()

        # ── Total loss ────────────────────────────────────────────────────────
        total_loss = str_loss + unc_loss + margin_loss
        return total_loss, loss_dict

    def _calc_loss_box(
            self,
            pred_noprompt_logits: torch.Tensor,
            pred_prompt_prob: torch.Tensor,
            **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_dict = {}

        # 1. Structure Loss
        str_loss = structure_loss(pred_noprompt_logits, pred_prompt_prob.detach())
        loss_dict['noprompt_box_str'] = str_loss.item()
        return str_loss, loss_dict

    def _calc_loss_point(self, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        pass


def generate_eval_data(loader: Any, is_multifold: bool, device: torch.device) -> Iterator[Tuple[Optional[torch.Tensor], ...]]:
    """统一验证集数据生成器"""
    if is_multifold:
        for item in loader:
            yield to_device(item[0], device)
    else:
        for _ in range(len(loader)):
            rgb, gt_pil, th, _, out_hw = loader.load_data()

            gt = torch.from_numpy(np.asarray(gt_pil, np.float32) / 255.0).unsqueeze(0).unsqueeze(0)

            pack = (rgb, th, gt, None, None, None, None, None, out_hw)

            yield to_device(pack, device)


def calculate_mae(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """计算单个样本的归一化平均绝对误差"""
    if pred.numel() != gt.numel():
        raise ValueError(f"Size mismatch: pred {pred.shape} vs gt {gt.shape}")

    res = pred.view(-1)
    gt = gt.view(-1)

    r_min, r_max = res.min(), res.max()
    if r_max > r_min:
        res = (res - r_min) / (r_max - r_min + 1e-8)
    else:
        res = torch.zeros_like(res)

    return torch.abs(res - gt).mean().item()


def save_checkpoint(model, epoch, mae, best_mae, best_epoch, save_path, filename):
    """保存模型"""
    if mae < best_mae or epoch == 1:
        best_mae = mae
        best_epoch = epoch
        torch.save(model.state_dict(), str(Path(save_path) / filename))
        return best_mae, best_epoch, True
    return best_mae, best_epoch, False














