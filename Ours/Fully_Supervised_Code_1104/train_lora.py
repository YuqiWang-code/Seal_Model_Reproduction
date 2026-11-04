"""训单模态视觉 LoRA（RGB 或 T），供全监督版冻结使用。

原始工程里的 ``outputs/RGB+LoRA`` 与 ``outputs/T+LoRA`` 权重没有随代码提供，这里
按同一思路重训：SAM3 图像模型 + 视觉 qkv/proj 上的 rank-16 LoRA（alpha=32），
**只训练 LoRA**，其余参数冻结；用 Seal train 的 GT 框作 Decoder 的几何提示，
用 GT mask 做结构损失 + Sigmoid Focal + Dice 监督；模型选择用 Seal val 的像素 IoU@0.5。

产出 ``<out>/lora_weights.pt``，格式为 ``{"state_dict": {含 lora 的参数}}``，
``lora_layers.load_lora_weights`` 可直接加载（唯一后缀匹配）。

用法：
    python train_lora.py --modality rgb --out ../weights/RGB+LoRA
    python train_lora.py --modality t   --out ../weights/T+LoRA
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("SEAL_CUDA", "1"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torchvision.transforms import v2  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "SAM3_LoRA"))

from sam3.model.data_misc import FindStage  # noqa: E402
from sam3.model.geometry_encoders import Prompt  # noqa: E402
from sam3.model_builder import build_sam3_image_model  # noqa: E402

from data import load_t_image, read_boxes  # noqa: E402
from lora_layers import LoRAConfig, LoRALinear, apply_lora_to_model  # noqa: E402
from train import (  # noqa: E402
    dice_loss,
    iou_counts,
    dataset_iou,
    prediction_probability,
    sigmoid_focal_loss,
    upsample_logits,
)
from utils import structure_loss  # noqa: E402

MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


class ModalityDataset(Dataset):
    """单模态图像 + GT 框 + GT mask。T 走逐图 min-max，RGB 直接读。"""

    def __init__(self, split_root, modality: str, image_size: int = 1008, load_gt: bool = True):
        self.root = Path(split_root)
        self.modality = modality
        self.image_size = image_size
        self.load_gt = load_gt
        img_dir = self.root / ("RGB" if modality == "rgb" else "T")
        ext = ".jpg" if modality == "rgb" else ".tif"
        stems = sorted(p.stem for p in img_dir.glob(f"*{ext}"))
        if not stems:
            raise FileNotFoundError(f"{img_dir} 下没有 {ext}")
        self.items = []
        for stem in stems:
            box_path = self.root / "box" / f"{stem}.txt"
            gt_path = self.root / "GT" / f"{stem}.png"
            if not box_path.is_file() or (load_gt and not gt_path.is_file()):
                raise FileNotFoundError(f"{stem}: 缺 box 或 GT")
            self.items.append((stem, img_dir / f"{stem}{ext}", box_path, gt_path if load_gt else None))
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((image_size, image_size), antialias=True),
            v2.Normalize(mean=MEAN, std=STD),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        stem, img_path, box_path, gt_path = self.items[index]
        if self.modality == "rgb":
            with Image.open(img_path) as im:
                image = im.convert("RGB")
            w, h = image.size
        else:
            image = load_t_image(img_path)
            w, h = image.size
        tensor = self.transform(image)
        boxes = read_boxes(box_path, w, h, self.image_size)
        out = {"stem": stem, "image": tensor, "boxes": torch.tensor(boxes, dtype=torch.float32)}
        if self.load_gt:
            with Image.open(gt_path) as im:
                out["gt_mask"] = torch.from_numpy(np.array(im.convert("L")) > 128)
        return out


def collate_single(batch):
    return batch[0]


def make_find_stage(device):
    return FindStage(
        img_ids=torch.tensor([0], device=device, dtype=torch.long),
        text_ids=torch.tensor([0], device=device, dtype=torch.long),
        input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
        input_points=None, input_points_mask=None,
    )


def clone_tree(obj):
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if isinstance(obj, dict):
        return {k: clone_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clone_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(clone_tree(v) for v in obj)
    return obj


@torch.no_grad()
def encode_visual_text(model, device):
    return model.backbone.forward_text(["visual"], device=device)


def grounding(model, image, boxes_cxcywh, text_outputs, find_stage):
    out = model.backbone.forward_image(image)
    features = clone_tree(out)
    for key, value in text_outputs.items():
        features[key] = clone_tree(value)
    prompt = Prompt(box_embeddings=boxes_cxcywh.to(torch.float32).view(-1, 1, 4))
    return model.forward_grounding(
        backbone_out=features, find_input=find_stage, find_target=None, geometric_prompt=prompt
    )


def merge_topk(outputs, top_k: int) -> torch.Tensor:
    scores = outputs["pred_logits"].sigmoid()
    presence = outputs.get("presence_logit_dec")
    if presence is not None:
        scores = scores * presence.sigmoid().unsqueeze(1)
    query_scores = scores[0].squeeze(-1)
    masks = outputs["pred_masks"][0]
    k = min(int(top_k), int(query_scores.numel()))
    index = torch.topk(query_scores, k=k).indices
    return masks.index_select(0, index).max(dim=0).values


def main():
    ap = argparse.ArgumentParser(description="训练单模态视觉 LoRA（RGB / T）")
    ap.add_argument("--modality", choices=("rgb", "t"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default="/share_datasets/Seal_Model_Reproduction/Ours/weights/sam3.pt")
    ap.add_argument("--train_data_dir", default="/share_datasets/Seal_Model_Reproduction/Seal_data/train")
    ap.add_argument("--val_data_dir", default="/share_datasets/Seal_Model_Reproduction/Seal_data/val")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--img_size", type=int, default=1008)
    ap.add_argument("--val_interval", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    device = torch.device("cuda")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_lora.log"

    def log(message):
        line = f"{datetime.now()} {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    bpe = HERE / "SAM3_LoRA" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    log(f"args={vars(args)}")
    model = build_sam3_image_model(
        bpe_path=str(bpe), device="cpu", eval_mode=True, checkpoint_path=args.checkpoint,
        load_from_HF=False, enable_segmentation=True, enable_inst_interactivity=False, compile=False,
    )
    cfg = LoRAConfig(rank=16, alpha=32.0, dropout=0.0, target_modules=["qkv", "proj"],
                     apply_to_vision_encoder=True, apply_to_text_encoder=False,
                     apply_to_geometry_encoder=False, apply_to_detr_encoder=False,
                     apply_to_detr_decoder=False, apply_to_mask_decoder=False)
    model = apply_lora_to_model(model, cfg)
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, LoRALinear):
            for param in module.lora.parameters():
                param.requires_grad = True
    lora_params = [p for p in model.parameters() if p.requires_grad]
    log(f"trainable LoRA tensors={sum(1 for _ in lora_params)} params(M)={sum(p.numel() for p in lora_params)/1e6:.3f}")
    model.to(device)
    model.eval()
    for module in model.backbone.vision_backbone.trunk.modules():
        if module.__class__.__name__ == "DropPath":
            module.eval()

    train_ds = ModalityDataset(args.train_data_dir, args.modality, args.img_size, load_gt=True)
    val_ds = ModalityDataset(args.val_data_dir, args.modality, args.img_size, load_gt=True)
    if args.limit:
        train_ds.items = train_ds.items[: args.limit]
        val_ds.items = val_ds.items[: max(1, args.limit // 4)]
    log(f"train={len(train_ds)} val={len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=args.num_workers,
                              collate_fn=collate_single, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
                            collate_fn=collate_single, pin_memory=True)

    text_outputs = encode_visual_text(model, device)
    find_stage = make_find_stage(device)
    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=args.weight_decay)

    best = {"iou": -1.0, "epoch": 0}
    for epoch in range(1, args.epochs + 1):
        model.eval()
        total, steps = 0.0, 0
        for step, batch in enumerate(train_loader, start=1):
            image = batch["image"].unsqueeze(0).to(device, non_blocking=True)
            boxes = batch["boxes"].to(device)
            gt_mask = batch["gt_mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = grounding(model, image, boxes, text_outputs, find_stage)
                logits = merge_topk(out, int(boxes.shape[0]))
                pred_map = upsample_logits(logits, image.shape[-2:])
                gt = F.interpolate(gt_mask.float()[None, None], size=image.shape[-2:], mode="nearest")
                loss = structure_loss(pred_map, gt) + sigmoid_focal_loss(pred_map, gt) + dice_loss(pred_map, gt)
            loss.backward()
            optimizer.step()
            total += float(loss)
            steps = step
            if step % args.log_every == 0 or step == len(train_loader):
                log(f"Epoch[{epoch:02d}/{args.epochs:02d}] Step[{step:04d}/{len(train_loader):04d}] loss={float(loss):.6f}")
        log(f"#TRAIN# epoch={epoch} mean_loss={total/max(steps,1):.6f}")

        if epoch % args.val_interval == 0 or epoch == args.epochs:
            tp = fp = fn = 0
            n = 0
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                for batch in val_loader:
                    image = batch["image"].unsqueeze(0).to(device)
                    boxes = batch["boxes"].to(device)
                    gt_mask = batch["gt_mask"].to(device)
                    out = grounding(model, image, boxes, text_outputs, find_stage)
                    logits = merge_topk(out, int(boxes.shape[0]))
                    prob = prediction_probability(logits, gt_mask)
                    a, b, c = iou_counts(prob > 0.5, gt_mask)
                    tp, fp, fn = tp + a, fp + b, fn + c
                    n += 1
            iou = dataset_iou(tp, fp, fn)
            log(f"#VAL# epoch={epoch} iou={iou:.6f} samples={n}")
            if iou > best["iou"]:
                best = {"iou": iou, "epoch": epoch}
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora" in k}
                torch.save({"state_dict": state, "epoch": epoch, "iou": iou, "modality": args.modality},
                           out_dir / "lora_weights.pt")
                log(f"保存 {out_dir/'lora_weights.pt'} (epoch={epoch} iou={iou:.6f})")

    log(f"结束：best_iou={best['iou']:.6f} @ epoch {best['epoch']}")


if __name__ == "__main__":
    main()
