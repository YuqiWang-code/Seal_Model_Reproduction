"""Adapt SSFam RGBT to Seal using sparse labels derived from training boxes."""

import argparse
import ast
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "代码" / "RGBT"))
from SAM_Model.model import Model  # noqa: E402
from lscloss import LocalSaliencyCoherence  # noqa: E402
from smoothness import smoothness_loss  # noqa: E402


def index_images(directory):
    return {path.stem: path for path in directory.iterdir() if path.is_file()}


def load_thermal(path):
    with Image.open(path) as image:
        if image.mode.startswith("I;16") or image.mode == "I":
            values = np.asarray(image, dtype=np.float32)
            lo, hi = float(values.min()), float(values.max())
            if hi > lo:
                values = (values - lo) * (255.0 / (hi - lo))
            else:
                values = np.zeros_like(values)
            return Image.fromarray(np.rint(values).astype(np.uint8)).convert("RGB")
        return image.convert("RGB")


RGB_TRANSFORM = transforms.Compose((
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
))
THERMAL_TRANSFORM = transforms.Compose((transforms.Resize((1024, 1024)), transforms.ToTensor()))
GRAY_TRANSFORM = transforms.Compose((transforms.Resize((1024, 1024)), transforms.ToTensor()))


class SealBoxes(Dataset):
    def __init__(self, seal_dir):
        root = seal_dir / "train"
        self.rgb = index_images(root / "RGB")
        self.thermal = index_images(root / "T")
        self.box_dir = root / "box"
        self.names = sorted(self.rgb)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        with Image.open(self.rgb[name]) as source:
            rgb = source.convert("RGB")
            width, height = rgb.size
            image = RGB_TRANSFORM(rgb)
            gray = GRAY_TRANSFORM(rgb.convert("L"))
        thermal = THERMAL_TRANSFORM(load_thermal(self.thermal[name]))
        boxes = np.loadtxt(self.box_dir / f"{name}.txt", dtype=np.float32).reshape(-1, 4)
        centers = np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2))
        rng = np.random.default_rng(index)
        background = []
        while len(background) < 5 * len(centers):
            x, y = rng.integers(0, width), rng.integers(0, height)
            if not np.any((x >= boxes[:, 0]) & (x <= boxes[:, 2]) & (y >= boxes[:, 1]) & (y <= boxes[:, 3])):
                background.append((x, y))
        points = np.concatenate((centers, np.asarray(background, dtype=np.float32)))
        points[:, 0] = np.clip(points[:, 0] * 1024 / width, 0, 1023)
        points[:, 1] = np.clip(points[:, 1] * 1024 / height, 0, 1023)
        labels = np.concatenate((np.ones(len(centers)), np.zeros(len(background))))
        return image, thermal, gray, torch.as_tensor(points, dtype=torch.float32), torch.as_tensor(labels, dtype=torch.float32)


def structure_loss(pred, target):
    weight = 1 + 5 * torch.abs(F.avg_pool2d(target, 31, stride=1, padding=15) - target)
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    bce = (weight * bce).sum((2, 3)) / weight.sum((2, 3))
    probability = pred.sigmoid()
    intersection = (probability * target * weight).sum((2, 3))
    union = ((probability + target) * weight).sum((2, 3))
    iou = 1 - (intersection + 1) / (union - intersection + 1)
    return (bce + iou).mean()


def validate(model, seal_dir, output_dir, evaluator):
    root = seal_dir / "val"
    rgb = index_images(root / "RGB")
    thermal = index_images(root / "T")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.inference_mode():
        for gt in sorted((root / "GT").glob("*.png")):
            with Image.open(rgb[gt.stem]) as source:
                image = RGB_TRANSFORM(source.convert("RGB")).unsqueeze(0).cuda()
            depth = THERMAL_TRANSFORM(load_thermal(thermal[gt.stem])).unsqueeze(0).cuda()
            logits, _ = model(image, depth, None)
            with Image.open(gt) as target:
                size = (target.height, target.width)
            prediction = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
            prediction = prediction.sigmoid().cpu().numpy().squeeze()
            prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min() + 1e-8)
            cv2.imwrite(str(output_dir / gt.name), np.rint(prediction * 255).astype(np.uint8))
    metrics_file = output_dir.parent / f"{output_dir.name}_metrics.json"
    subprocess.run(
        (sys.executable, str(evaluator), "--pred_root", str(output_dir), "--gt_dir", str(root / "GT"),
         "--json_output", str(metrics_file)),
        capture_output=True, text=True, check=True,
    )
    return json.loads(metrics_file.read_text())["Seal"]["F"]


def metric_functions(path):
    source = path.read_text(encoding="utf-8")
    names = {"s_measure_folder", "weighted_f_measure_folder", "e_measure_folder", "mae_folder"}
    return {
        node.name: ast.get_source_segment(source, node)
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seal-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = ROOT.parent / "evaluate.py"
    if metric_functions(evaluator) != metric_functions(ROOT.parent / "Ours" / "evaluate.py"):
        raise ValueError("Evaluation functions differ from Ours/evaluate.py")
    evaluator_source = args.output_dir / "evaluate_source.py"
    shutil.copyfile(evaluator, evaluator_source)
    (args.output_dir / "config.json").write_text(json.dumps({
        "source_checkpoint": str(args.checkpoint),
        "train_inputs": "Seal_data/train/{RGB,T,box}",
        "validation_inputs": "Seal_data/val/{RGB,T,GT}",
        "thermal": "per-image 16-bit min-max to 8-bit RGB",
        "foreground": "one box center per box",
        "background": "five points outside all boxes per box",
        "input_size": 1024,
        "batch_size": 1,
        "optimizer": "Adam",
        "learning_rate": 5e-5,
        "epochs": args.epochs,
        "validation_interval": 5,
        "selection_metric": "Ours weighted_f_measure_folder",
    }, indent=2))

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    cfg = argparse.Namespace(model_type="vit_l", checkpoint=None,
                             freeze_image_encoder=True, freeze_prompt_encoder=True,
                             freeze_mask_decoder=False)
    model = Model(cfg)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.setup()
    model.cuda()
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=5e-5)
    loader = DataLoader(SealBoxes(args.seal_dir), batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    smooth = smoothness_loss().cuda()
    coherence = LocalSaliencyCoherence().cuda()
    best_fw = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for step, (image, thermal, gray, points, labels) in enumerate(loader, 1):
            image, thermal, gray = image.cuda(), thermal.cuda(), gray.cuda()
            points, labels = points.cuda(), labels.cuda()
            optimizer.zero_grad(set_to_none=True)
            student, teacher = model(image, thermal, (points, labels.long()))
            xy = points[0].long()
            selected = teacher[0, 0, xy[:, 1], xy[:, 0]]
            sparse_loss = 0.5 * (
                F.binary_cross_entropy_with_logits(selected[labels[0] == 1], labels[0][labels[0] == 1])
                + F.binary_cross_entropy_with_logits(selected[labels[0] == 0], labels[0][labels[0] == 0])
            )
            teacher_prob = teacher.sigmoid()
            rgb_small = F.interpolate(image, scale_factor=0.25, mode="bilinear", align_corners=True)
            prob_small = F.interpolate(teacher_prob, scale_factor=0.25, mode="bilinear", align_corners=True)
            local_loss = coherence(prob_small, [{"weight": 1, "xy": 6, "rgb": 0.1}], 5,
                                   {"rgb": rgb_small}, 256, 256)["loss"]
            loss = structure_loss(student, teacher_prob.detach()) + sparse_loss + 0.3 * smooth(teacher_prob, gray) + local_loss
            loss.backward()
            optimizer.step()
            total += loss.item()
            if step % 100 == 0 or step == len(loader):
                print(f"epoch {epoch}/{args.epochs} step {step}/{len(loader)} loss {total / step:.5f}", flush=True)

        if epoch % 5 == 0:
            checkpoint = args.output_dir / f"epoch_{epoch:02d}.pth"
            maps = args.output_dir / f"val_maps_epoch_{epoch:02d}"
            torch.save(model.state_dict(), checkpoint)
            fw = float(validate(model, args.seal_dir, maps, evaluator_source))
            with (args.output_dir / "validation.jsonl").open("a", encoding="utf-8") as records:
                records.write(json.dumps({"epoch": epoch, "val_Fw": fw,
                                          "checkpoint": checkpoint.name,
                                          "prediction_dir": maps.name,
                                          "prediction_count": len(list(maps.glob("*.png")))}) + "\n")
            print(f"epoch {epoch} val Fw {fw:.8f}", flush=True)
            if fw > best_fw:
                best_fw = fw
                (args.output_dir / "selection.json").write_text(json.dumps({
                    "epoch": epoch, "val_Fw": fw, "checkpoint": checkpoint.name,
                    "prediction_dir": maps.name,
                }, indent=2))
                print(f"best epoch {epoch} val Fw {fw:.8f}", flush=True)

    selected = json.loads((args.output_dir / "selection.json").read_text())
    subprocess.run((sys.executable, str(ROOT / "infer_seal.py"), "--modality", "RGBT",
                    "--checkpoint", str(args.output_dir / selected["checkpoint"]),
                    "--seal-dir", str(args.seal_dir),
                    "--output-dir", str(args.output_dir / "test_maps")), check=True)
    subprocess.run((sys.executable, str(evaluator_source), "--pred_root", str(args.output_dir / "test_maps"),
                    "--gt_dir", str(args.seal_dir / "test" / "GT"),
                    "--json_output", str(args.output_dir / "test_metrics.json")), check=True)


if __name__ == "__main__":
    main()
