from __future__ import annotations
import torch
import torch.utils.data as data
import torchvision.transforms as T
from PIL import Image
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from utils import open_gray, open_rgb, open_thermal, preprocess_sam3_image, read_bbox_txt, list_images_unique_by_stem, align_modalities_by_stem_strict, permuted_kfold_indices


# Constants
RGB_EXTS = frozenset({".jpg"})
THERMAL_EXTS = frozenset({".tif"})
GT_EXTS = frozenset({".png"})
BBOX_EXTS = frozenset({".txt"})

SEAL_RGB_DIRS = ("RGB",)
SEAL_THERMAL_DIRS = ("T",)
SEAL_BBOX_DIRS = ("box", "bbox")
SEAL_GT_DIRS = ("GT",)


def _single_item_collate(batch):
    return batch[0]


def _resolve_existing_dir(root: Union[str, Path], candidates: Tuple[str, ...], required: bool = True) -> Path:
    root = Path(root)
    if root.is_dir():
        for name in candidates:
            candidate = root / name
            if candidate.is_dir():
                return candidate
    if required:
        raise FileNotFoundError(f"Could not resolve a dataset directory under {root}. Tried: {candidates}")
    return root


def resolve_seal_split_paths(split_root: Union[str, Path]) -> Dict[str, Path]:
    split_root = Path(split_root)
    rgb_root = _resolve_existing_dir(split_root, SEAL_RGB_DIRS)
    thermal_root = _resolve_existing_dir(split_root, SEAL_THERMAL_DIRS)
    bbox_root = _resolve_existing_dir(split_root, SEAL_BBOX_DIRS, required=False)
    gt_root = _resolve_existing_dir(split_root, SEAL_GT_DIRS, required=False)

    paths = {
        "rgb": rgb_root,
        "t": thermal_root,
    }
    if bbox_root.exists() and bbox_root.is_dir() and bbox_root != split_root:
        paths["bbox"] = bbox_root
    if gt_root.exists() and gt_root.is_dir() and gt_root != split_root:
        paths["gt"] = gt_root
    return paths


@dataclass(frozen=True)
class KFoldConfig:
    k_folds: int = 5
    fold_idx: Optional[int] = None
    train_mode: bool = True
    seed: int = 2025


@dataclass(frozen=True)
class Sam3PreprocessConfig:
    image_size: int = 1008
    mean: float = 0.5
    std: float = 0.5


@dataclass(frozen=True)
class DatasetConfig:
    trainsize: int
    kfold: KFoldConfig = field(default_factory=KFoldConfig)
    sam3: Sam3PreprocessConfig = field(default_factory=Sam3PreprocessConfig)


class SalObjDataset(data.Dataset):
    def __init__(
            self,
            image_root: Union[str, Path],
            thermal_root: Union[str, Path],
            bbox_root: Union[str, Path],
            cfg: DatasetConfig,
    ) -> None:
        super().__init__()
        self.cfg = cfg

        roots_config = {
            "rgb": (image_root, RGB_EXTS),
            "t": (thermal_root, THERMAL_EXTS),
            "box": (bbox_root, BBOX_EXTS),
        }

        file_maps = {k: list_images_unique_by_stem(path, exts=exts) for k, (path, exts) in roots_config.items()}

        aligned = align_modalities_by_stem_strict(file_maps)
        self.data_paths = aligned

        total_len = len(aligned["rgb"])
        if cfg.kfold.fold_idx is not None:
            train_idx, val_idx = permuted_kfold_indices(total_len, k_folds=cfg.kfold.k_folds, fold_idx=cfg.kfold.fold_idx, seed=cfg.kfold.seed)
            indices = train_idx if cfg.kfold.train_mode else val_idx
            for key in self.data_paths:
                self.data_paths[key] = [self.data_paths[key][i] for i in indices]

        self.size = len(self.data_paths["rgb"])

        self.resize = T.Resize((cfg.trainsize, cfg.trainsize))
        self.up = T.Resize((cfg.trainsize, cfg.trainsize), interpolation=T.InterpolationMode.NEAREST)
        self.to_tensor = T.ToTensor()

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        paths = {k: v[index] for k, v in self.data_paths.items()}

        images = {
            "rgb": open_rgb(paths["rgb"]),
            "t": open_thermal(paths["t"]),
        }

        boxes_abs = self._load_box_prompt(images, paths)

        tensors = self._transform_images(images)
        rgb_sam3, th_sam3 = self._preprocess_for_sam3(tensors["rgb"], tensors["t"])

        out_hw = torch.tensor([self.cfg.trainsize, self.cfg.trainsize], dtype=torch.int64)

        return (
            rgb_sam3,
            th_sam3,
            boxes_abs,
            out_hw,
        )

    def _load_box_prompt(self, images: Dict[str, Image.Image], paths: Dict[str, Path]) -> torch.Tensor:
        target_size = self.cfg.trainsize
        boxes_raw = read_bbox_txt(paths["box"])
        w, h = images["rgb"].size
        scale_x = target_size / float(w)
        scale_y = target_size / float(h)

        boxes_abs = boxes_raw.clone()
        boxes_abs[:, [0, 2]] *= scale_x
        boxes_abs[:, [1, 3]] *= scale_y
        return boxes_abs

    def _transform_images(self, images: Dict[str, Image.Image]) -> Dict[str, torch.Tensor]:
        """统一进行 Resize 和 ToTensor。"""
        tensors = {}
        for key, img in images.items():
            resized = self.resize(img)
            tensors[key] = self.to_tensor(resized)
        return tensors

    def _preprocess_for_sam3(self, rgb: torch.Tensor, th: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """为 SAM3 模型进行标准化预处理。"""
        sam_cfg = self.cfg.sam3

        def _prep(t):
            return preprocess_sam3_image(t, image_size=sam_cfg.image_size, mean=sam_cfg.mean, std=sam_cfg.std)

        return _prep(rgb), _prep(th)


class NamedSalObjDataset(SalObjDataset):
    def __getitem__(self, index):
        data_pack = super().__getitem__(index)
        name = self.data_paths["rgb"][index].stem
        return data_pack, name


class TestDataset:
    def __init__(
            self,
            image_root: Union[str, Path],
            gt_root: Union[str, Path],
            thermal_root: Union[str, Path],
            *,
            testsize: int,
            sam3_image_size: int = 1008,
    ):
        self.testsize = testsize
        self.sam3_cfg = Sam3PreprocessConfig(image_size=sam3_image_size)

        roots = {
            "rgb": (image_root, RGB_EXTS),
            "gt": (gt_root, GT_EXTS),
            "t": (thermal_root, THERMAL_EXTS),
        }

        file_maps = {k: list_images_unique_by_stem(path, exts=exts) for k, (path, exts) in roots.items()}

        self.data_paths = align_modalities_by_stem_strict(file_maps)

        self.resize = T.Resize((self.testsize, self.testsize))
        self.to_tensor = T.ToTensor()
        self.length = len(self.data_paths["rgb"])
        self.idx = 0

    def __len__(self) -> int:
        return self.length

    def load_data(self):
        idx = self.idx % self.length
        paths = {k: v[idx] for k, v in self.data_paths.items()}

        rgb_pil = open_rgb(paths["rgb"])
        th_pil = open_thermal(paths["t"])
        gt_pil = open_gray(paths["gt"])
        name = paths["gt"].name

        # Transform
        rgb_ts = self.to_tensor(self.resize(rgb_pil))
        th_ts = self.to_tensor(self.resize(th_pil))

        # SAM3 Preprocess
        def _prep(x):
            return preprocess_sam3_image(x, image_size=self.sam3_cfg.image_size, mean=self.sam3_cfg.mean, std=self.sam3_cfg.std).unsqueeze(0)

        rgb_sam3 = _prep(rgb_ts)
        th_sam3 = _prep(th_ts)

        out_hw = torch.tensor([self.testsize, self.testsize], dtype=torch.int64).unsqueeze(0)

        self.idx += 1
        return rgb_sam3, gt_pil, th_sam3, name, out_hw


def get_loader(
        image_root: Union[str, Path],
        thermal_root: Union[str, Path],
        bbox_root: Union[str, Path],
        *,
        batchsize: int = 1,
        trainsize: int,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        fold_idx: Optional[int] = None,
        k_folds: int = 5,
        train_mode: bool = True,
        seed: int = 2025,
        sam3_image_size: int = 1008,
        require_names: bool = False,
) -> data.DataLoader:
    if batchsize != 1:
        raise ValueError(f"Batchsize must be 1. Got {batchsize}.")

    cfg = DatasetConfig(
        trainsize=trainsize,
        kfold=KFoldConfig(
            k_folds=k_folds,
            fold_idx=fold_idx,
            train_mode=train_mode,
            seed=seed
        ),
        sam3=Sam3PreprocessConfig(image_size=sam3_image_size),
    )

    DatasetClass = NamedSalObjDataset if require_names else SalObjDataset

    ds = DatasetClass(
        image_root=image_root,
        thermal_root=thermal_root,
        bbox_root=bbox_root,
        cfg=cfg,
    )

    return data.DataLoader(
        ds,
        batch_size=batchsize,
        shuffle=shuffle if train_mode else False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=train_mode,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=_single_item_collate,
    )
