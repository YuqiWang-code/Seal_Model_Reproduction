"""RGB-T 框监督数据集。

目录约定与 demo8 / demo9 相同：

    <split>/RGB/*.jpg
    <split>/T/*.tif
    <split>/box/*.txt
    <split>/GT/*.png   # 仅验证集

图像预处理与这两份训练脚本一致，使冻结的视觉 LoRA 看到原来的分布。
框在读取时转成归一化 cxcywh。batch size 固定为 1。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import v2


RGB_EXTS = frozenset({".jpg"})
THERMAL_EXTS = frozenset({".tif"})
BOX_EXTS = frozenset({".txt"})
GT_EXTS = frozenset({".png"})


def _list_by_stem(root: Path, exts: frozenset) -> Dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"缺少目录: {root}")

    found: Dict[str, Path] = {}
    for path in root.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in exts:
            continue
        stem = path.stem
        if stem in found:
            logging.warning("主干 %s 重复，保留 %s，忽略 %s", stem, found[stem], path)
            continue
        found[stem] = path
    if not found:
        raise FileNotFoundError(f"目录中没有可用文件: {root}")
    return found


def _align(maps: Dict[str, Dict[str, Path]]) -> List[str]:
    stem_sets = {name: set(items) for name, items in maps.items()}
    common = set.intersection(*stem_sets.values())
    if not common:
        names = ", ".join(maps)
        raise ValueError(f"模态之间没有共同文件名: {names}")

    for name, stems in stem_sets.items():
        extra = stems - common
        if extra:
            examples = ", ".join(sorted(extra)[:5])
            raise ValueError(
                f"模态 {name} 的文件名与其他模态不一致，多出 {len(extra)} 个，例如: {examples}"
            )
    return sorted(common)


def load_t_image(image_path: Path) -> Image.Image:
    """Seal 的 16 位热红外：按单图 min-max 拉伸到 8 位，再转三通道 RGB。

    原包的写法是 ``x - min/(max-min)*255``，只适用于 8 位图；Seal 的 ``T`` 是
    16 位 TIFF（实测 uint16、约 26,400–28,500），该式会得到 2 万多再转 uint8
    回绕成噪声。这里按全项目统一口径改为逐图 min-max（见 AGENTS.md），
    输入分布与“先把 T 转 8 位再做实验”的原始流程等价。
    """
    with Image.open(image_path) as t_image:
        t_array = np.asarray(t_image)
    if t_array.ndim == 3:
        t_array = t_array[..., 0]
    t_array = t_array.astype(np.float32)
    low, high = float(t_array.min()), float(t_array.max())
    if high > low:
        t_array = (t_array - low) * (255.0 / (high - low))
    else:
        t_array = np.zeros_like(t_array)
    t_array = np.clip(np.rint(t_array), 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(t_array, mode="L").convert("RGB")


def _open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
    return height, width


def _load_gt_mask(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.array(image.convert("L"))
    return torch.from_numpy(array > 128)


def _discard_box(path: Path, line_no: int, detail: str) -> None:
    message = f"丢弃非法框 {path} 第 {line_no} 行: {detail}"
    print(message)
    logging.warning(message)


def read_boxes(path: Path, orig_w: int, orig_h: int, image_size: int) -> tuple:
    """读取像素 xyxy，缩放到 image_size 后转为归一化 cxcywh。"""
    boxes = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 4:
                _discard_box(path, line_no, line)
                continue
            try:
                x1, y1, x2, y2 = (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
            except ValueError:
                _discard_box(path, line_no, line)
                continue
            if x2 <= x1 or y2 <= y1:
                _discard_box(path, line_no, line)
                continue

            scale_x = image_size / float(orig_w)
            scale_y = image_size / float(orig_h)
            x1_r = min(max(x1 * scale_x, 0.0), float(image_size))
            y1_r = min(max(y1 * scale_y, 0.0), float(image_size))
            x2_r = min(max(x2 * scale_x, 0.0), float(image_size))
            y2_r = min(max(y2 * scale_y, 0.0), float(image_size))
            if x2_r <= x1_r or y2_r <= y1_r:
                _discard_box(path, line_no, f"缩放到 {image_size} 后为空")
                continue

            boxes.append(
                (
                    ((x1_r + x2_r) / 2.0) / image_size,
                    ((y1_r + y2_r) / 2.0) / image_size,
                    (x2_r - x1_r) / image_size,
                    (y2_r - y1_r) / image_size,
                )
            )

    if not boxes:
        raise ValueError(f"没有合法框: {path}")
    return tuple(boxes)


class RGBTBoxDataset(data.Dataset):
    """一张图一对 RGB / T，外加该图的全部真实框。"""

    def __init__(
        self,
        split_root: Union[str, Path],
        image_size: int = 1008,
        load_gt: bool = False,
    ) -> None:
        super().__init__()
        split_root = Path(split_root)
        self.image_size = image_size
        self.load_gt = load_gt

        maps = {
            "rgb": _list_by_stem(split_root / "RGB", RGB_EXTS),
            "t": _list_by_stem(split_root / "T", THERMAL_EXTS),
            "box": _list_by_stem(split_root / "box", BOX_EXTS),
        }
        if load_gt:
            maps["gt"] = _list_by_stem(split_root / "GT", GT_EXTS)
        stems = _align(maps)

        self.samples: List[dict] = []
        for stem in stems:
            rgb_path = maps["rgb"][stem]
            orig_h, orig_w = _image_size(rgb_path)
            sample = {
                "stem": stem,
                "rgb": rgb_path,
                "t": maps["t"][stem],
                "boxes": read_boxes(maps["box"][stem], orig_w, orig_h, image_size),
                "orig_size": (orig_h, orig_w),
            }
            if load_gt:
                sample["gt"] = maps["gt"][stem]
            self.samples.append(sample)

        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _resize(self, image: Image.Image) -> Image.Image:
        return image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        rgb = self.transform(self._resize(_open_rgb(sample["rgb"])))
        thermal = self.transform(self._resize(load_t_image(sample["t"])))
        item = {
            "rgb": rgb,
            "t": thermal,
            "boxes": torch.tensor(sample["boxes"], dtype=torch.float32),
            "stem": sample["stem"],
        }
        if self.load_gt:
            item["orig_size"] = sample["orig_size"]
            item["gt_mask"] = _load_gt_mask(sample["gt"])
        return item


def _single_item_collate(batch):
    return batch[0]


def get_loader(
    split_root: Union[str, Path],
    *,
    image_size: int,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    load_gt: bool = False,
) -> data.DataLoader:
    if batch_size != 1:
        raise ValueError(f"batch size 必须为 1，收到 {batch_size}")

    dataset = RGBTBoxDataset(split_root, image_size=image_size, load_gt=load_gt)
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=_single_item_collate,
    )
