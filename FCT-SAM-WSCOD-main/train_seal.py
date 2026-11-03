"""Train FCT-SAM on Seal weak labels and evaluate the validation-selected model."""

import argparse
import ast
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import loss
from dataset import img_normalize
from model.TestCamoFormer import CamoFormer
from model.decoder.decoder_baseline import Decoder
from utils.tools import Crop, Flip, Translate, SaliencyStructureConsistency
from prepare_seal import check_pseudo_labels
from prepare_seal_box_sam import selected_files, validate_box_labels

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from evaluate import mae_folder


def snapshot_evaluation(run):
    source = (PROJECT / 'evaluate.py').read_text(encoding='utf-8-sig')
    reference = (PROJECT / 'Ours/evaluate.py').read_text(encoding='utf-8-sig')
    names = {'s_measure_folder', 'weighted_f_measure_folder', 'e_measure_folder', 'mae_folder'}
    def functions(text):
        return {node.name: ast.get_source_segment(text, node) for node in ast.parse(text).body
                if isinstance(node, ast.FunctionDef) and node.name in names}
    if functions(source) != functions(reference) or len(functions(source)) != 4:
        raise ValueError('Root evaluation functions differ from Ours/evaluate.py')
    shutil.copyfile(PROJECT / 'evaluate.py', run / 'evaluate_source.py')
    shutil.copyfile(PROJECT / 'metrics.py', run / 'small_target_metrics_source.py')
    shutil.copyfile(PROJECT / 'evaluate_small_targets.py', run / 'evaluate_small_targets_source.py')


def check_prediction_names(predictions, gt):
    actual = {p.name for p in predictions.glob('*.png')}
    expected = {p.name for p in gt.glob('*.png')}
    if actual != expected:
        raise ValueError(f'Prediction pairing: missing={len(expected-actual)}, extra={len(actual-expected)}')
    return len(actual)


class SealDataset(Dataset):
    def __init__(self, data, split, size, pseudo=None):
        self.files = sorted((data / split / "RGB").glob("*"))
        self.labels = pseudo if pseudo is not None else data / split / "GT"
        self.size = size
        self.training = pseudo is not None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255
        label = np.asarray(Image.open(self.labels / (path.stem + ".png")).convert("L")) > 128
        image = torch.from_numpy(img_normalize(image)).permute(2, 0, 1)
        label = torch.from_numpy(label.astype(np.float32))[None]
        image = F.interpolate(image[None], (self.size, self.size), mode="bilinear", align_corners=True)[0]
        if self.training:
            label = F.interpolate(label[None], (self.size, self.size), mode="nearest")[0]
            choice = random.randrange(4)
            if choice < 3:
                dims = [(-1,), (-2,), (-2, -1)][choice]
                image, label = image.flip(dims), label.flip(dims)
        return image, label, path.stem


def build_model(checkpoint):
    model = CamoFormer(cfg=None)
    model.decoder = Decoder(64)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=True)
    return model.cuda()


@torch.inference_mode()
def predict(model, data, output=None):
    model.eval()
    total = 0.0
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
    for image, gt, name in DataLoader(data, batch_size=1, num_workers=2):
        prediction = model(image.cuda())[-1]
        prediction = F.interpolate(prediction, gt.shape[-2:], mode="bilinear", align_corners=False)
        prediction = prediction.sigmoid().cpu().numpy().squeeze()
        prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min() + 1e-8)
        total += float(np.abs(prediction - gt.numpy().squeeze()).mean())
        if output is not None:
            Image.fromarray((prediction * 255).astype(np.uint8)).save(output / (name[0] + ".png"))
    return total / len(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pseudo", type=Path, required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--resume", type=Path)
    parser.add_argument('--supervision', choices=['sparse_points', 'box_sam'], default='sparse_points')
    args = parser.parse_args()
    if args.supervision == 'box_sam':
        invalid = validate_box_labels(args.data, args.pseudo, selected_files(args.data))
        generation = json.loads((args.pseudo / 'generation.json').read_text(encoding='utf-8'))
        if generation['supervision'] != 'Box' or generation['uses_gt_masks']:
            raise ValueError('Expected box-only SAM pseudo labels')
    else:
        prompts = json.loads((args.pseudo / 'prompts.json').read_text(encoding='utf-8'))
        invalid = check_pseudo_labels(sorted((args.data / 'train/RGB').glob('*')), args.pseudo, prompts)
    if invalid:
        raise ValueError(f"Training pseudo labels fail annotation consistency: {len(invalid)} invalid/missing images")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(8)
    args.run.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(batch_size=2, accumulation=2, encoder_lr=1e-5, decoder_lr=1e-4,
                  seed=42, selection='minimum original mae_folder on saved validation PNGs, every 5 epochs',
                  initialization='full COD checkpoint; encoder and decoder trainable')
    if args.supervision == 'sparse_points':
        config.update(foreground_points=1, background_points=5)
    (args.run / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    snapshot_evaluation(args.run)
    model = build_model(args.init)
    optimizer = torch.optim.Adam([
        {"params": model.encoder.parameters(), "lr": 1e-5},
        {"params": model.decoder.parameters(), "lr": 1e-4},
    ])
    train = SealDataset(args.data, "train", args.size, args.pseudo)
    validation = SealDataset(args.data, "val", args.size)
    loader = DataLoader(train, batch_size=2, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    first, best = 0, float("inf")
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        first, best = saved["epoch"], saved["best_val_mae"]
    for epoch in range(first, args.epochs):
        started = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for step, (image, target, _) in enumerate(loader):
            image, target = image.cuda(non_blocking=True), target.cuda(non_blocking=True)
            transform = random.choice([Flip(random.randrange(2)), Translate(0.15), Crop(0.7, 0.7)])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(image)
                smaller = F.interpolate(transform(image), scale_factor=0.5, mode="bilinear", align_corners=True)
                other = model(smaller)
            outputs = tuple(x.float() for x in outputs)
            other = tuple(x.float() for x in other)
            target_other = F.interpolate(transform(target), scale_factor=0.5, mode="nearest")
            reference = F.interpolate(transform(outputs[-1]), scale_factor=0.5,
                                      mode="bilinear", align_corners=True)
            supervised, _ = loss.multi_bce(outputs, target)
            supervised_other, _ = loss.multi_bce(other, target_other)
            consistency = SaliencyStructureConsistency(other[-1], reference.detach(), 0.85)
            objective = supervised + supervised_other + consistency
            if not torch.isfinite(objective):
                raise RuntimeError(f"Non-finite objective at epoch {epoch + 1}, step {step + 1}")
            (objective / 2).backward()
            if (step + 1) % 2 == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            running += objective.item()
            if (step + 1) % 50 == 0:
                print(f"epoch={epoch + 1} step={step + 1}/{len(loader)} loss={running / (step + 1):.6f}", flush=True)
        if len(loader) % 2:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        val_mae = None
        if (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
            candidate_dir = args.run / 'validation' / f'epoch_{epoch + 1:03d}'
            predict(model, validation, candidate_dir)
            count = check_prediction_names(candidate_dir, args.data / 'val/GT')
            val_mae = float(mae_folder(str(candidate_dir), str(args.data / 'val/GT'))['mean_MAE'])
            checkpoint = {'state_dict': model.state_dict(), 'epoch': epoch + 1,
                          'val_mae': val_mae, 'input_size': args.size,
                          'decoder': 'decoder_baseline', 'supervision': args.supervision}
            candidate = args.run / f'epoch_{epoch + 1:03d}.pth'
            torch.save(checkpoint, candidate)
            with (args.run / 'validation.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps({'epoch': epoch + 1, 'val_mae': val_mae,
                                         'n_images': count, 'checkpoint': str(candidate),
                                         'predictions': str(candidate_dir)}) + '\n')
            if val_mae < best:
                best = val_mae
                shutil.copyfile(candidate, args.run / 'best.pth')
                (args.run / 'selection.json').write_text(json.dumps({
                    'epoch': epoch + 1, 'val_mae': best, 'criterion': 'minimum original mae_folder',
                    'checkpoint': str(candidate), 'n_validation_images': count}, indent=2), encoding='utf-8')
        torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1, "best_val_mae": best}, args.run / "last.pth")
        entry = {"epoch": epoch + 1, "loss": running / len(loader), "val_mae": val_mae,
                 "best_val_mae": best if np.isfinite(best) else None, "seconds": time.time() - started}
        with (args.run / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)
    saved = torch.load(args.run / "best.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(saved["state_dict"])
    output = args.run / "test_maps"
    predict(model, SealDataset(args.data, "test", args.size), output)
    count = check_prediction_names(output, args.data / 'test/GT')
    if count != 1983:
        raise ValueError(f'Expected 1983 Seal test predictions, found {count}')
    result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "evaluate.py"),
                             "--pred_root", str(output), "--gt_dir", str(args.data / "test/GT"),
                             "--json_output", str(args.run / "test_metrics_original_full.json")],
                            check=True, stdout=subprocess.PIPE, text=True)
    (args.run / "test_metrics.json").write_text(result.stdout, encoding="utf-8")
    print(result.stdout, flush=True)
    small_output = args.run / 'test_small_targets.json'
    subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / 'evaluate_small_targets.py'),
                    '--pred_root', str(output), '--gt_dir', str(args.data / 'test/GT'),
                    '--json_output', str(small_output)], check=True)
    original = json.loads((args.run / 'test_metrics_original_full.json').read_text(encoding='utf-8'))['Seal']
    small = json.loads(small_output.read_text(encoding='utf-8'))
    combined = {'checkpoint': str(args.run / 'best.pth'), 'epoch': saved['epoch'],
                'supervision': args.supervision, 'input_size': args.size, 'n_images': count,
                'val_mae': saved['val_mae'], 'prediction_dir': str(output), 'metrics': {
                    'S': original['S'], 'Fw': original['F'], 'E': original['E'], 'MAE': original['MAE'],
                    **small['metrics']}}
    (args.run / 'test_eight_metrics.json').write_text(json.dumps(combined, indent=2), encoding='utf-8')


if __name__ == "__main__":
    main()
