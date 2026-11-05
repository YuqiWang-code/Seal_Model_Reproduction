"""框监督训练用到的损失、指标、日志和分布式工具。"""

from __future__ import annotations

import logging
import os
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0


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


def sigmoid_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = FOCAL_ALPHA,
    gamma: float = FOCAL_GAMMA,
) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = prob * target + (1.0 - prob) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    return (alpha_t * (1.0 - p_t).pow(gamma) * ce).mean()


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = tuple(range(1, prob.dim()))
    inter = (prob * target).sum(dim=dims)
    denom = prob.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def iou_counts(prob: torch.Tensor, mask: torch.Tensor) -> tuple[int, int, int]:
    """与 metrics.AS1000Metrics 相同：概率 > 0.5 后统计像素 TP / FP / FN。"""
    binary = prob.detach().reshape(-1) > 0.5
    mask = mask.detach().reshape(-1).to(device=binary.device).bool()
    tp = binary & mask
    fp = binary & ~mask
    fn = ~binary & mask
    return int(tp.sum().item()), int(fp.sum().item()), int(fn.sum().item())


def dataset_iou(tp: int, fp: int, fn: int) -> float:
    """与 metrics.AS1000Metrics.get 的 IoU 相同。"""
    union = tp + fp + fn
    return tp / union if union else 0.0


def format_metric(value: float, places: int = 5) -> str:
    """四舍五入到指定小数位。"""
    quant = Decimal("1").scaleb(-places)
    number = Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP)
    return f"{number:.{places}f}"


def upsample_logits(logit: torch.Tensor, size: tuple) -> torch.Tensor:
    return F.interpolate(logit.float()[None, None], size=size, mode="bilinear", align_corners=False)


def prediction_probability(logit: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
    """预测概率图缩放到 GT 分辨率，数值限制在 [0, 1]。"""
    probability = torch.sigmoid(logit.float())[None, None]
    probability = F.interpolate(probability, size=tuple(gt_mask.shape[-2:]), mode="bilinear", align_corners=False)[0, 0]
    return probability.clamp(0.0, 1.0)


def count_trainable(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def is_main_process() -> bool:
    return not distributed() or dist.get_rank() == 0


def _cpu_collectives() -> bool:
    return dist.get_backend() != "nccl"


def all_reduce_(tensor: torch.Tensor, op: dist.ReduceOp) -> None:
    if not _cpu_collectives():
        dist.all_reduce(tensor, op=op)
        return
    cpu = tensor.detach().to("cpu")
    dist.all_reduce(cpu, op=op)
    tensor.copy_(cpu)


def _broadcast_(tensor: torch.Tensor, src: int = 0) -> None:
    if not _cpu_collectives():
        dist.broadcast(tensor, src=src)
        return
    cpu = tensor.detach().to("cpu")
    dist.broadcast(cpu, src=src)
    tensor.copy_(cpu)


def broadcast_module(module: nn.Module | None) -> None:
    """让各卡从同一份 M1 / LoRA3 初始权重开始。"""
    if module is None or not distributed():
        return
    for tensor in list(module.parameters()) + list(module.buffers()):
        _broadcast_(tensor.data)


def sync_gradients(params: list[nn.Parameter]) -> None:
    """各卡各自反传后，把可训练参数的梯度求平均。

    LoRA3 挂在冻结的 SAM3 里面，前向不经过一个统一的 DDP 模块，所以这里手动同步。
    """
    if not distributed() or not params:
        return
    flags = torch.tensor(
        [0 if param.grad is None else 1 for param in params],
        device=params[0].device,
        dtype=torch.int32,
    )
    all_reduce_(flags, dist.ReduceOp.SUM)
    grads: list[torch.Tensor] = []
    for param, count in zip(params, flags.tolist()):
        if int(count) == 0:
            continue
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        grads.append(param.grad)
    if not grads:
        return

    world_size = dist.get_world_size()
    buckets: dict[tuple[torch.dtype, str], list[torch.Tensor]] = {}
    for grad in grads:
        buckets.setdefault((grad.dtype, str(grad.device)), []).append(grad)
    for bucket in buckets.values():
        flat = torch.cat([grad.reshape(-1) for grad in bucket])
        all_reduce_(flat, dist.ReduceOp.SUM)
        flat.div_(world_size)
        offset = 0
        for grad in bucket:
            count = grad.numel()
            grad.copy_(flat[offset:offset + count].view_as(grad))
            offset += count


def init_distributed() -> tuple[int, int, torch.device]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        if not torch.cuda.is_available():
            raise RuntimeError("需要 CUDA。")
        torch.cuda.set_device(0)
        return 0, 1, torch.device("cuda", 0)

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("多卡请用 torchrun 启动，例如 torchrun --nproc_per_node=4 train.py")
    local_rank = int(os.environ["LOCAL_RANK"])
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA。")
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(f"LOCAL_RANK={local_rank} 超出可见 GPU 数量 {torch.cuda.device_count()}。")
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        backend = "nccl" if dist.is_nccl_available() else "gloo"
        dist.init_process_group(backend=backend)
    return rank, world_size, torch.device("cuda", local_rank)


def require_equal_steps(loader, device: torch.device) -> None:
    if not distributed():
        return
    local = torch.tensor([len(loader)], device=device, dtype=torch.int64)
    minimum = local.clone()
    maximum = local.clone()
    all_reduce_(minimum, dist.ReduceOp.MIN)
    all_reduce_(maximum, dist.ReduceOp.MAX)
    if int(minimum.item()) != int(maximum.item()):
        raise RuntimeError(f"各卡训练步数不一致: min={int(minimum.item())} max={int(maximum.item())}")
