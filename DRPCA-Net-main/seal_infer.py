"""Sliding-window inference helper for the fully-supervised IRSTD baselines.

Native 256x256 windows, half-overlap, averaged logits, then sigmoid to an 8-bit
PNG at the original size. `channels` selects single- or three-channel input.
"""
import numpy as np
import torch
import torch.nn.functional as F


def _windows(length, window, stride):
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


@torch.no_grad()
def sliding_window_logits(model, image, window=256, stride=128, batch=8,
                          device='cuda', output_fn=None, forward_fn=None):
    """image: float32 CHW tensor already normalized. Returns HxW logits numpy."""
    _, h, w = image.shape
    ys, xs = _windows(h, window, stride), _windows(w, window, stride)
    patches, coords = [], []
    for y in ys:
        for x in xs:
            patch = image[:, y:y + window, x:x + window]
            if patch.shape[1] != window or patch.shape[2] != window:
                patch = F.pad(patch, (0, window - patch.shape[2], 0, window - patch.shape[1]))
            patches.append(patch)
            coords.append((y, x))
    acc = torch.zeros((h, w), dtype=torch.float32, device=device)
    cnt = torch.zeros((h, w), dtype=torch.float32, device=device)
    for start in range(0, len(patches), batch):
        chunk = torch.stack(patches[start:start + batch]).to(device)
        raw = forward_fn(model, chunk) if forward_fn else model(chunk)
        logits = output_fn(raw) if output_fn else raw
        logits = logits[:, 0]
        for i, (y, x) in enumerate(coords[start:start + batch]):
            y1 = min(y + window, h)
            x1 = min(x + window, w)
            acc[y:y1, x:x1] += logits[i, :y1 - y, :x1 - x]
            cnt[y:y1, x:x1] += 1.0
    return (acc / cnt.clamp_min(1e-6)).cpu().numpy()


def logits_to_uint8(logits):
    prob = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    return np.clip(np.round(prob * 255.0), 0, 255).astype(np.uint8)
