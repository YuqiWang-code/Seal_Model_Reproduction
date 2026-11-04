import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("SEAL_CUDA", "1"))
import argparse
import re
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from typing import List, Tuple
from pathlib import Path
from sam3.sam3model import Model
from data import TestDataset, resolve_seal_split_paths


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mode", type=str, default="single", choices=["single", "multi"], help="运行模式：single=单折；multi=多折")
    parser.add_argument("--checkpoint", type=Path, default='train_logs/best3.pth', help="单折权重路径")
    parser.add_argument("--train_logs", type=Path, default='train_logs', help="多折权重目录")
    parser.add_argument("--img_size", type=int, default=1008)
    parser.add_argument("--save_dir", type=Path, default="test_maps")
    parser.add_argument("--test_data_dir", type=Path, default=Path("/share_datasets/Seal_Model_Reproduction/Seal_data/test"), help="Seal 测试集根目录")
    parser.add_argument("--t_encoder_type", type=str, default="resnet50", choices=["resnet50", "lora"])
    parser.add_argument("--resnet_ckpt", type=str, default="/share_datasets/Seal_Model_Reproduction/Ours/weights/resnet50-19c8e357.pth")
    return parser.parse_args()


_FOLD_RE = re.compile(r"^best-fold(\d+)\.pth$", re.IGNORECASE)


def discover_fold_ckpts(root: Path) -> List[Tuple[int, Path]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Train logs 目录不存在: {root}")

    ckpts = []

    for p in root.glob("best-fold*.pth"):
        if not p.is_file():
            continue

        if m := _FOLD_RE.match(p.name):
            fid = int(m.group(1))
            ckpts.append((fid, p))

    if not ckpts:
        raise FileNotFoundError(f"{root} 下未找到符合 best-fold*.pth 命名规则的文件")

    return sorted(ckpts, key=lambda x: x[0])


@torch.no_grad()
def infer_dataset(model: torch.nn.Module, dataset_root: Path, out_dir: Path, img_size: int):
    """对单个数据集进行推理"""
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = resolve_seal_split_paths(dataset_root)
    loader = TestDataset(
        image_root=str(paths["rgb"]),
        gt_root=str(paths["gt"]),
        thermal_root=str(paths["t"]),
        testsize=img_size,
        sam3_image_size=img_size,
    )

    print(f"    Dataset: {dataset_root.name} | Samples: {len(loader)}")

    for _ in range(len(loader)):
        rgb_sam3, gt_pil, th_sam3, name, _out_hw = loader.load_data()

        rgb_sam3 = rgb_sam3.cuda(non_blocking=True)
        th_sam3 = th_sam3.cuda(non_blocking=True)

        gt = np.array(gt_pil, dtype=np.float32)
        gt /= (gt.max() + 1e-8)

        mask_1, _mask_2, _noprompt_feat, _prompt_feat = model(rgb_sam3, th_sam3, None)

        pred = F.interpolate(mask_1, size=gt.shape, mode="bilinear", align_corners=False)
        pred = pred.sigmoid().cpu().numpy().squeeze()

        p_min, p_max = pred.min(), pred.max()
        pred = (pred - p_min) / (p_max - p_min + 1e-8)

        save_path = out_dir / name
        cv2.imwrite(str(save_path), (pred * 255).round().astype(np.uint8))


def test_one_fold(ckpt_path: Path, args):
    print(f"\n=== Testing checkpoint ===")
    print(f"Checkpoint: {ckpt_path}")

    model = Model(ckpt_path=None, image_size=args.img_size, t_encoder_type=args.t_encoder_type,
                  resnet_ckpt=args.resnet_ckpt).cuda()
    state_dict = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    save_root = args.save_dir / ckpt_path.stem

    print(f"  -> Inferring Seal test...")
    infer_dataset(model, args.test_data_dir, save_root / args.test_data_dir.name, args.img_size)

    print(f"  -> Results saved to: {save_root}")


def main():
    opt = parse_args()
    mode = opt.mode.lower()

    if mode == "single":
        if not opt.checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint 不存在: {opt.checkpoint}")
        test_one_fold(opt.checkpoint, opt)

    elif mode == "multi":
        print(f"Scanning folds in {opt.train_logs}...")
        try:
            fold_ckpts = discover_fold_ckpts(opt.train_logs)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return

        print(f"Found {len(fold_ckpts)} checkpoints: {[p.name for _, p in fold_ckpts]}")

        for fid, p in fold_ckpts:
            test_one_fold(p, opt)

    print("\nAll tasks completed.")


if __name__ == "__main__":
    main()
