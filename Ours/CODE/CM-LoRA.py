"""RGB LoRA、T LoRA 和共享 LoRA3。

两套模态 LoRA 按 demo8 / demo9 注入视觉主干的 qkv 与 proj，然后冻结。
LoRA3 挂在同一批线性层上，参数由两套模型共享，不注册进 SAM3。
"""
from __future__ import annotations
import sys
import importlib.resources
import pkg_resources
import torch
import torch.nn as nn
from pathlib import Path
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
from sam3.model_builder import build_sam3_image_model
from lora_layers import LoRAConfig, LoRALayer, LoRALinear, apply_lora_to_model, load_lora_weights


LORA3_RANK = 8
LORA3_ALPHA = 16.0
LORA3_DROPOUT = 0.05
LORA3_LAYER_COUNT = 64
_VISION_LORA_TARGETS = ("qkv", "proj")


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"找不到 {label}: {path}")


def _vision_lora_config() -> LoRAConfig:
    return LoRAConfig(
        rank=16,
        alpha=32.0,
        dropout=0.0,
        target_modules=list(_VISION_LORA_TARGETS),
        apply_to_vision_encoder=True,
        apply_to_text_encoder=False,
        apply_to_geometry_encoder=False,
        apply_to_detr_encoder=False,
        apply_to_detr_decoder=False,
        apply_to_mask_decoder=False,
    )


def _linear_shape(layer: LoRALinear) -> tuple[int, int]:
    return (layer.in_features, layer.out_features)


def configure_low_memory_model(model: nn.Module) -> None:
    """关闭 DAC，与 demo8 训练时的 Decoder 设置一致。query 数已是 200。"""
    decoder = model.transformer.decoder
    decoder.dac = False
    decoder.num_o2m_queries = 0
    model.dac = False


def _load_sam3(checkpoint_path: Path) -> nn.Module:
    bpe_path = SAM3_PKG_ROOT / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    _require_file(bpe_path, "SAM3 词表")
    _require_file(checkpoint_path, "SAM3 权重")
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
    return model


def _freeze_model(model: nn.Module, device: torch.device) -> nn.Module:
    for name, param in model.backbone.named_parameters():
        if "lora" not in name:
            param.data = param.data.to(torch.bfloat16)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    model.to(device)
    return model


def build_frozen_base_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """与 demo7 相同，只加载冻结的原始 SAM3，不注入 LoRA。"""
    return _freeze_model(_load_sam3(checkpoint_path), device)


def build_frozen_lora_model(checkpoint_path: Path, lora_path: Path, device: torch.device) -> nn.Module:
    """按 demo10 加载一套视觉 LoRA，然后连同 LoRA 一起冻结。"""
    _require_file(lora_path, "LoRA 权重")
    model = _load_sam3(checkpoint_path)
    model = apply_lora_to_model(model, _vision_lora_config())
    load_lora_weights(model, str(lora_path))
    return _freeze_model(model, device)


class SharedLoRA(nn.Module):
    """两套视觉主干共用的 LoRA3。按模块全名索引到各自的 LoRALayer。"""

    def __init__(self, named_shapes: dict[str, tuple[int, int]]) -> None:
        super().__init__()
        self.index: dict[str, int] = {}
        layers: list[LoRALayer] = []
        for index, (name, (in_features, out_features)) in enumerate(named_shapes.items()):
            self.index[name] = index
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
    return {
        name: module
        for name, module in model.named_modules()
        if "vision_backbone" in name
        and name.split(".")[-1] in _VISION_LORA_TARGETS
        and isinstance(module, LoRALinear)
    }


def _bind_shared_lora(layer: LoRALinear, shared: SharedLoRA, name: str) -> None:
    object.__setattr__(layer, "shared_lora", shared)
    object.__setattr__(layer, "shared_name", name)


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
        rgb_shape = _linear_shape(rgb_sites[name])
        t_shape = _linear_shape(t_sites[name])
        if rgb_shape != t_shape:
            raise RuntimeError(f"{name} 两侧线性层形状不一致：{rgb_shape} 与 {t_shape}。")
        named_shapes[name] = rgb_shape

    shared = SharedLoRA(named_shapes)
    for name in named_shapes:
        _bind_shared_lora(rgb_sites[name], shared, name)
        _bind_shared_lora(t_sites[name], shared, name)
    return shared


def enable_trunk_checkpoint(model: nn.Module) -> None:
    """打开 ViT trunk 的 activation checkpoint，并关掉 DropPath。"""
    trunk = model.backbone.vision_backbone.trunk
    trunk.train()
    for module in trunk.modules():
        if module.__class__.__name__ == "DropPath":
            module.eval()
