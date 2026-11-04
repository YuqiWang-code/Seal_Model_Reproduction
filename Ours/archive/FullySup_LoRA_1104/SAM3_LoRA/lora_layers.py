"""LoRA 注入与权重加载（供 ``Fully_Supervised_Code_1104/train.py`` 使用）。

仓库里没有原始的 ``lora_layers`` 模块，这里按 ``sam3/adapter.py`` 里
``LoRALayer``/``DualLoRALinear`` 的数学写法重写一套等价实现：

* 每个视觉 ``qkv`` / ``proj`` 线性层外包一层 ``LoRALinear``，内部保持一个
  **模态 LoRA**（rank/alpha 可配，默认 16/32），其数学与 ``sam3.adapter.LoRA``
  相同：``up(down(dropout(x))) * alpha / rank``，down 用 kaiming 初始化、
  up 用零初始化，因此在加载预训练权重之前该分支是恒等映射之外的零增量。
* ``LoRALinear`` 另外支持挂一个**共享 LoRA3**（``shared_lora`` + ``shared_name``），
  由 ``train.py`` 的 ``build_shared_lora`` 在两套模型之间共享；前向时在模态
  LoRA 之外再加 ``shared_lora.delta(name, x)``。

对外接口与 ``train.py`` 的调用一一对应：``LoRAConfig``、``LoRALayer``、
``LoRALinear``、``apply_lora_to_model``、``load_lora_weights``。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn


@dataclass
class LoRAConfig:
    """与 ``train.py`` 中 ``LoRAConfig(...)`` 的调用参数一致。"""

    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    target_modules: Sequence[str] = field(default_factory=lambda: ("qkv", "proj"))
    apply_to_vision_encoder: bool = True
    apply_to_text_encoder: bool = False
    apply_to_geometry_encoder: bool = False
    apply_to_detr_encoder: bool = False
    apply_to_detr_decoder: bool = False
    apply_to_mask_decoder: bool = False


class LoRALayer(nn.Module):
    """低秩增量：``up(down(dropout(x))) * alpha / rank``。"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank 必须为正，收到 {rank}")
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(self.dropout(x))) * self.scaling


class LoRALinear(nn.Module):
    """``base`` 冻结，模态 LoRA 可训；可选叠加共享 LoRA3。"""

    def __init__(self, base: nn.Linear, cfg: LoRAConfig, name: str = "") -> None:
        super().__init__()
        self.name = name
        self.base = base
        self.base.requires_grad_(False)
        self.lora = LoRALayer(
            in_features=base.in_features,
            out_features=base.out_features,
            rank=cfg.rank,
            alpha=cfg.alpha,
            dropout=cfg.dropout,
        )
        # 由 train.py 的 build_shared_lora 用 object.__setattr__ 挂上；不注册成子模块，
        # 这样共享 LoRA3 的参数只出现在它自己的 ModuleList 里，不会被 SAM3 state_dict 收录。
        self.shared_lora = None
        self.shared_name: Optional[str] = None

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x) + self.lora(x)
        shared = getattr(self, "shared_lora", None)
        if shared is not None:
            out = out + shared.delta(self.shared_name, x)
        return out


def _in_scope(name: str, cfg: LoRAConfig) -> bool:
    scopes = (
        ("vision_backbone", cfg.apply_to_vision_encoder),
        ("text_encoder", cfg.apply_to_text_encoder),
        ("text", cfg.apply_to_text_encoder),
        ("geometry_encoder", cfg.apply_to_geometry_encoder),
        ("transformer.encoder", cfg.apply_to_detr_encoder),
        ("transformer.decoder", cfg.apply_to_detr_decoder),
        ("mask_decoder", cfg.apply_to_mask_decoder),
    )
    return any(marker in name and enabled for marker, enabled in scopes)


def apply_lora_to_model(model: nn.Module, cfg: LoRAConfig) -> nn.Module:
    """把视觉主干里 ``target_modules`` 指定的 ``nn.Linear`` 换成 ``LoRALinear``。"""
    targets: list[tuple[str, nn.Module, str, nn.Linear]] = []
    for name, module in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf not in cfg.target_modules:
            continue
        if not _in_scope(name, cfg):
            continue
        if isinstance(module, LoRALinear):
            continue
        if isinstance(module, nn.Linear):
            parent, _, attr = name.rpartition(".")
            targets.append((name, model.get_submodule(parent) if parent else model, attr, module))

    for name, parent, attr, linear in targets:
        setattr(parent, attr, LoRALinear(linear, cfg, name=name))
    if not targets:
        raise RuntimeError("没有匹配到可注入 LoRA 的线性层，请检查 target_modules 与作用域设置。")
    return model


def _extract_state_dict(obj) -> dict:
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "lora", "lora_state_dict"):
            value = obj.get(key)
            if isinstance(value, dict):
                return value
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise ValueError("LoRA 权重文件不是可识别的 state dict")


def _normalize_key(key: str) -> str:
    key = key.replace("module.", "", 1) if key.startswith("module.") else key
    # 去掉 LoRA 注入前的旧前缀写法，便于按后缀匹配。
    return key


def load_lora_weights(model: nn.Module, path: str, strict_suffix: bool = True) -> int:
    """把 ``path`` 里的 LoRA 参数装到 ``model`` 上，返回成功匹配的张量数。

    兼容两种常见格式：只存 LoRA 的 state dict，或整模型的 state dict。
    先按原键名精确匹配；剩余参数用“唯一后缀”匹配：从长到短尝试键名末尾若干段，
    只在目标里该后缀唯一时才采用，这样 ``module.`` / ``backbone.`` 之类前缀差异
    既能匹配，也不会把 64 个 qkv/proj 站点混在一起。
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = _extract_state_dict(payload)

    own = {k: v for k, v in model.state_dict().items() if "lora" in k}
    candidates = {_normalize_key(k): v for k, v in state.items() if isinstance(v, torch.Tensor)}
    matched: dict[str, torch.Tensor] = {}

    for key in list(candidates):
        if key in own and own[key].shape == candidates[key].shape:
            matched[key] = candidates.pop(key)

    if strict_suffix and candidates:
        # 后缀 -> 唯一目标键；重复后缀不进入映射，避免错配。
        unique_suffix: dict[str, str] = {}
        ambiguous: set[str] = set()
        for depth in range(10, 3, -1):
            for target in own:
                suffix = ".".join(target.split(".")[-depth:])
                if suffix in unique_suffix:
                    ambiguous.add(suffix)
                else:
                    unique_suffix[suffix] = target
        by_suffix = {s: t for s, t in unique_suffix.items() if s not in ambiguous}

        for key in list(candidates):
            parts = key.split(".")
            for depth in range(min(10, len(parts)), 3, -1):
                suffix = ".".join(parts[-depth:])
                target = by_suffix.get(suffix)
                if target is not None and own[target].shape == candidates[key].shape:
                    matched[target] = candidates.pop(key)
                    break

    if not matched:
        raise RuntimeError(f"{path} 中没有能与模型 LoRA 参数匹配的张量")

    model.load_state_dict(matched, strict=False)
    print(f"[lora] {path}: 匹配 {len(matched)}/{len(own)} 个张量，未用 {len(candidates)} 个")
    return len(matched)


def count_lora_parameters(model: nn.Module) -> int:
    return sum(p.numel() for name, p in model.named_parameters() if "lora" in name)


def iter_lora_modules(model: nn.Module) -> Iterable[LoRALinear]:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            yield module
