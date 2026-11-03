"""Generate SAM pseudo labels from fixed sparse prompts on Seal train images."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry


def agrees_with_prompts(mask, xy, labels):
    points = np.rint(xy).astype(int)
    points[:, 0] = np.clip(points[:, 0], 0, mask.shape[1] - 1)
    points[:, 1] = np.clip(points[:, 1], 0, mask.shape[0] - 1)
    return bool(np.array_equal(mask[points[:, 1], points[:, 0]], np.asarray(labels, dtype=bool)))


def choose_consistent_mask(masks, scores, xy, labels):
    valid = [index for index, mask in enumerate(masks) if agrees_with_prompts(mask, xy, labels)]
    if not valid:
        return None
    return masks[max(valid, key=lambda index: scores[index])]


def restore_masks(masks, rotation, flip, width, height):
    restored = []
    for mask in masks:
        mask = cv2.resize(mask.astype(np.uint8), (768, 768), interpolation=cv2.INTER_NEAREST)
        if flip:
            mask = np.flip(mask, axis=1)
        mask = np.rot90(mask, -rotation)
        restored.append(cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST) > 0)
    return np.stack(restored)


def label_response_filter(masks):
    if not masks:
        return None
    response = np.stack(masks).mean(axis=0)
    nonzero = response[response > 0]
    if not nonzero.size:
        return None
    return response >= float(nonzero.mean() / 4)


def check_pseudo_labels(files, folder, prompts):
    invalid = []
    for path in files:
        target = folder / (path.stem + '.png')
        if not target.exists():
            invalid.append({'name': path.stem, 'reason': 'missing'})
            continue
        mask = np.asarray(Image.open(target).convert('L')) > 128
        prompt = prompts[path.stem]
        if not agrees_with_prompts(mask, prompt['xy'], prompt['labels']):
            invalid.append({'name': path.stem, 'reason': 'prompt_inconsistent'})
    return invalid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--sam", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    files = sorted((args.data / "train/RGB").glob("*"))
    prompts = {}
    for index, path in enumerate(files):
        rng = np.random.default_rng(42 + index)
        gt = np.asarray(Image.open(args.data / "train/GT" / (path.stem + ".png"))) > 128
        fg = np.argwhere(gt)
        bg = np.argwhere(~gt)
        if not len(fg):
            raise ValueError(f"No foreground available for the positive prompt: {path.name}")
        points = np.concatenate((fg[rng.choice(len(fg), 1)], bg[rng.choice(len(bg), 5, replace=False)]))
        prompts[path.stem] = {"xy": points[:, ::-1].tolist(), "labels": [1, 0, 0, 0, 0, 0]}
    (args.output / "prompts.json").write_text(json.dumps(prompts, indent=2), encoding="utf-8")
    existing = [path for path in files if (args.output / (path.stem + '.png')).exists()]
    invalid_existing = check_pseudo_labels(existing, args.output, prompts)
    if invalid_existing:
        raise ValueError(f"{len(invalid_existing)} existing pseudo labels violate prompts; use a separate output directory")
    if all((args.output / (path.stem + ".png")).exists() for path in files):
        print(f"SAM pseudo labels ready: {len(files)}/{len(files)}", flush=True)
        return
    sam = sam_model_registry["vit_h"](checkpoint=str(args.sam)).cuda().eval()
    predictor = SamPredictor(sam)
    with torch.inference_mode():
        for index, path in enumerate(files):
            destination = args.output / (path.stem + ".png")
            if destination.exists():
                continue
            image = np.asarray(Image.open(path).convert("RGB"))
            height, width = image.shape[:2]
            base = cv2.resize(image, (768, 768), interpolation=cv2.INTER_LINEAR)
            original_xy = np.asarray(prompts[path.stem]["xy"], dtype=np.float32)
            xy = original_xy * np.array([768 / width, 768 / height], dtype=np.float32)
            labels = np.asarray(prompts[path.stem]["labels"])
            rng = np.random.default_rng(1042 + index)
            accepted_masks = []
            for iteration in range(10):
                rotation = int(rng.integers(4))
                flip = bool(rng.integers(2))
                side = int(768 * rng.choice([0.5, 1.0, 2.0]))
                augmented = np.rot90(base, rotation)
                coords = xy.copy()
                for _ in range(rotation):
                    coords = np.stack((coords[:, 1], 767 - coords[:, 0]), axis=1)
                if flip:
                    augmented = np.flip(augmented, axis=1)
                    coords[:, 0] = 767 - coords[:, 0]
                augmented = cv2.resize(augmented, (side, side), interpolation=cv2.INTER_LINEAR)
                coords *= side / 768
                predictor.set_image(np.ascontiguousarray(augmented))
                _, scores, logits = predictor.predict(
                    point_coords=coords[:1], point_labels=labels[:1], multimask_output=True
                )
                masks, scores, _ = predictor.predict(
                    point_coords=coords, point_labels=labels,
                    mask_input=logits[np.argmax(scores)][None], multimask_output=True
                )
                restored = restore_masks(masks, rotation, flip, width, height)
                selected = choose_consistent_mask(restored, scores, original_xy, labels)
                if selected is None:
                    masks, scores, _ = predictor.predict(
                        point_coords=coords, point_labels=labels, multimask_output=True
                    )
                    restored = restore_masks(masks, rotation, flip, width, height)
                    selected = choose_consistent_mask(restored, scores, original_xy, labels)
                if selected is not None:
                    accepted_masks.append(selected)
            result = label_response_filter(accepted_masks)
            if result is None or not agrees_with_prompts(result, original_xy, labels):
                print(f"Rejected pseudo label: {path.stem}; no prompt-consistent filtered response", flush=True)
                continue
            Image.fromarray(result.astype(np.uint8) * 255).save(destination)
            if (index + 1) % 10 == 0 or index + 1 == len(files):
                print(f"SAM pseudo labels: {index + 1}/{len(files)}", flush=True)
    invalid = check_pseudo_labels(files, args.output, prompts)
    (args.output / 'label_audit.json').write_text(json.dumps({'images': len(files), 'invalid': invalid}, indent=2), encoding='utf-8')
    if invalid:
        raise ValueError(f"Pseudo-label preparation incomplete: {len(invalid)} invalid/missing labels; training stopped")


if __name__ == "__main__":
    main()
