"""框监督训练用到的损失、匹配和日志工具。"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def setup_logger(save_path: str, filename: str = "log.txt") -> None:
    Path(save_path).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(save_path, filename),
        format="[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]",
        level=logging.INFO,
        filemode="a",
        datefmt="%Y-%m-%d %I:%M:%S %p",
        encoding="utf-8",
        force=True,
    )


def adjust_lr(
    optimizer: torch.optim.Optimizer,
    init_lr: float,
    epoch: int,
    *,
    decay_rate: float = 0.1,
    decay_epoch: int = 30,
) -> float:
    step = epoch // decay_epoch
    lr = init_lr * (decay_rate**step)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def clone_tree(obj: Any) -> Any:
    """复制嵌套的 Tensor / list / dict，保留梯度。"""
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if isinstance(obj, dict):
        return {key: clone_tree(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [clone_tree(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(clone_tree(value) for value in obj)
    return obj


def structure_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """加权 BCE + 加权 IoU。pred 是 logits，mask 是 [0, 1] 的目标。"""
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred_prob = torch.sigmoid(pred)
    inter = ((pred_prob * mask) * weit).sum(dim=(2, 3))
    union = ((pred_prob + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, width, height = boxes.unbind(-1)
    return torch.stack(
        (cx - 0.5 * width, cy - 0.5 * height, cx + 0.5 * width, cy + 0.5 * height),
        dim=-1,
    )


def pairwise_giou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """xyxy 框的两两 GIoU，形状 (M, N)。"""
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-6)

    lt_c = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb_c = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    area_c = (rb_c[..., 0] - lt_c[..., 0]).clamp(min=0) * (rb_c[..., 1] - lt_c[..., 1]).clamp(min=0)
    return iou - (area_c - union) / area_c.clamp(min=1e-6)


def box_supervision_loss(
    pred_cxcywh: torch.Tensor,
    gt_cxcywh: torch.Tensor,
    conf_logits: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """用置信度参与匹配，让高置信度框的数量靠近真实框数。

    预测框是 (M, 4)，置信度 logits 是 (M,)，真实框是 (N, 4)，都用归一化 cxcywh。
    匹配代价是 xyxy 上的 L1 + (1 - GIoU) + (1 - 置信度)。配上的框监督为置信度 1，
    没配上的监督为 0，因此过阈值的框数会靠近 N。框回归只加在配上的框上。
    """
    if pred_cxcywh.dim() != 2 or pred_cxcywh.shape[-1] != 4:
        raise ValueError(f"预测框形状应为 (M, 4)，收到 {tuple(pred_cxcywh.shape)}")
    if conf_logits.dim() != 1 or conf_logits.shape[0] != pred_cxcywh.shape[0]:
        raise ValueError(f"置信度 logits 形状应为 ({pred_cxcywh.shape[0]},)，收到 {tuple(conf_logits.shape)}")
    if gt_cxcywh.dim() != 2 or gt_cxcywh.shape[-1] != 4:
        raise ValueError(f"真实框形状应为 (N, 4)，收到 {tuple(gt_cxcywh.shape)}")
    if pred_cxcywh.shape[0] == 0 or gt_cxcywh.shape[0] == 0:
        raise ValueError("预测框和真实框的数量都要大于 0")

    pred_xyxy = box_cxcywh_to_xyxy(pred_cxcywh)
    gt_xyxy = box_cxcywh_to_xyxy(gt_cxcywh)
    giou = pairwise_giou(pred_xyxy, gt_xyxy)
    l1 = (pred_xyxy[:, None, :] - gt_xyxy[None, :, :]).abs().sum(dim=-1)
    conf_prob = conf_logits.detach().sigmoid()
    cost = (l1 + (1.0 - giou) + (1.0 - conf_prob[:, None])).detach().float().cpu().numpy()
    row, col = linear_sum_assignment(cost)

    device = pred_cxcywh.device
    row_index = torch.as_tensor(row, device=device, dtype=torch.long)
    col_index = torch.as_tensor(col, device=device, dtype=torch.long)
    n_matched = row_index.numel()
    matched = torch.zeros(pred_cxcywh.shape[0], dtype=torch.bool, device=device)
    matched[row_index] = True
    loss_conf = F.binary_cross_entropy_with_logits(conf_logits[matched], torch.ones_like(conf_logits[matched]))
    if (~matched).any():
        loss_conf = loss_conf + F.binary_cross_entropy_with_logits(conf_logits[~matched], torch.zeros_like(conf_logits[~matched]))
    return {
        "loss_l1": l1[row_index, col_index].sum() / n_matched,
        "loss_giou": (1.0 - giou[row_index, col_index]).sum() / n_matched,
        "loss_conf": loss_conf,
    }


def calc_mae(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """与 evaluate.py 的 calc_mae 相同。pred 在 [0, 1]，gt 为二值。"""
    pred = pred.detach().float().reshape(-1)
    gt = gt.detach().bool().to(dtype=pred.dtype, device=pred.device).reshape(-1)
    return float((pred - gt).abs().mean())
