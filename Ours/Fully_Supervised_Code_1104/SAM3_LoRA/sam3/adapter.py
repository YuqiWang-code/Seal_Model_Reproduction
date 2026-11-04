from __future__ import annotations
import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Literal, Union


LayerNormOption = Literal["none", "in", "out"]
Mode = Literal["rgb", "t", "t_new"]
AdapterScalar = Union[float, Literal["learnable_scalar"]]


@dataclass(frozen=True)
class LoRAConfig:
    alpha: int = 32
    rank: int = 16
    drop_rate: float = 0.05


LORA_CFG = LoRAConfig()


class Adapter(nn.Module):
    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int,
        *,
        dropout: float = 0.0,
        adapter_scalar: AdapterScalar = 1.0,
        layernorm: LayerNormOption = "in",
    ) -> None:
        super().__init__()
        self.layernorm = layernorm
        self._ln = nn.LayerNorm(input_dim) if layernorm != "none" else nn.Identity()

        self.proj = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, input_dim),
        )

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(()))
        else:
            self.register_buffer("scale", torch.tensor(float(adapter_scalar)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layernorm == "in":
            x = self._ln(x)

        out = self.proj(x) * self.scale

        if self.layernorm == "out":
            out = self._ln(out)

        return out


class LoRA(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()

        self.dropout = nn.Dropout(LORA_CFG.drop_rate)
        self.down = nn.Linear(input_dim, LORA_CFG.rank, bias=False)
        self.up = nn.Linear(LORA_CFG.rank, output_dim, bias=False)
        self.scaling = LORA_CFG.alpha / LORA_CFG.rank

        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(self.dropout(x))) * self.scaling


class DualLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, *, mode: Mode = "rgb") -> None:
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)

        in_dim, out_dim = base.in_features, base.out_features
        self.lora = nn.ModuleDict({"rgb": LoRA(in_dim, out_dim), "t": LoRA(in_dim, out_dim),})
        self.mode = mode

    @property
    def mode(self) -> Mode:
        return self._mode

    @mode.setter
    def mode(self, value: Mode) -> None:
        if value not in ("rgb", "t"):
            raise ValueError(f"mode must be 'rgb' or 't', got {value}")
        self._mode = value

    def set_mode(self, mode: Mode) -> None:
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora[self.mode](x)
