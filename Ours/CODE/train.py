"""全监督 RGB-T 训练。

冻结 SAM3 和两套视觉 LoRA。共享 LoRA3 的构建在 CM-LoRA.py，M1 的融合在 BISF.py，默认一起训练。
M1 融合 RGB / T 的图像特征后，配上数据集里的真实框进一次 Decoder。
预测 mask 用数据集 GT 做结构损失、Sigmoid Focal Loss 和 Dice Loss。
传入 --no-lora3 时只更新 M1。传入 --no-lora 时 RGB、T 和 LoRA3 都不用，两路图像共用冻结的原始 SAM3。

单卡直接 python train.py，默认只用 0 号卡。多卡每张卡仍一次处理一张图，梯度在卡间取平均：
    torchrun --nproc_per_node=4 train.py
"""

from __future__ import annotations

import os

# 单进程沿用原来的单卡。torchrun 会预先写入 LOCAL_RANK，这时必须看见全部卡。
if "LOCAL_RANK" not in os.environ:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0, 1, 2")

import argparse
import importlib.util
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load_cm_lora():
    path = ROOT / "CM-LoRA.py"
    spec = importlib.util.spec_from_file_location("cm_lora", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_cm_lora = _load_cm_lora()
build_frozen_base_model = _cm_lora.build_frozen_base_model
build_frozen_lora_model = _cm_lora.build_frozen_lora_model
build_shared_lora = _cm_lora.build_shared_lora
enable_trunk_checkpoint = _cm_lora.enable_trunk_checkpoint

import torch
torch.multiprocessing.set_sharing_strategy("file_system")
import torch.distributed as dist
import torch.nn as nn

from sam3.model.data_misc import FindStage
from sam3.model.geometry_encoders import Prompt

from BISF import M1
from data import get_loader, resize_gt, set_loader_epoch
from utils import (
    adjust_lr,
    all_reduce_,
    broadcast_module,
    calc_mae,
    clone_tree,
    count_trainable,
    dataset_iou,
    dice_loss,
    distributed,
    format_metric,
    init_distributed,
    iou_counts,
    is_main_process,
    prediction_probability,
    require_equal_steps,
    setup_logger,
    sigmoid_focal_loss,
    structure_loss,
    sync_gradients,
    upsample_logits,
)


def require_trainable(m1: nn.Module) -> list:
    params = [param for param in m1.parameters() if param.requires_grad]
    if not params:
        message = "M1 没有可训练参数。"
        logging.error(message)
        raise RuntimeError(message)
    return params


def make_find_stage(device: torch.device) -> FindStage:
    return FindStage(
        img_ids=torch.tensor([0], device=device, dtype=torch.long),
        text_ids=torch.tensor([0], device=device, dtype=torch.long),
        input_boxes=None,
        input_boxes_mask=None,
        input_boxes_label=None,
        input_points=None,
        input_points_mask=None,
    )


def make_geometric_prompt(boxes_cxcywh: torch.Tensor) -> Prompt:
    """正框，归一化 cxcywh，形状 (N, 1, 4)。标签和 padding mask 由 Prompt 补齐。"""
    if boxes_cxcywh.dim() != 2 or boxes_cxcywh.shape[-1] != 4:
        raise RuntimeError(f"框形状应为 (N, 4)，收到 {tuple(boxes_cxcywh.shape)}")
    count = boxes_cxcywh.shape[0]
    if count < 1:
        raise RuntimeError("至少需要一个框才能调用 Decoder。")
    boxes = boxes_cxcywh.to(dtype=torch.float32).view(count, 1, 4)
    return Prompt(box_embeddings=boxes)


def branch_backbone(fused: dict, text_outputs: dict) -> dict:
    """复制融合特征和文本。文本在整个训练里复用，不能被 Decoder 原地改写。"""
    features = clone_tree(fused)
    for key, value in text_outputs.items():
        features[key] = clone_tree(value)
    return features


def decode(model: nn.Module, features: dict, boxes_cxcywh: torch.Tensor, find_stage: FindStage) -> dict:
    if model.training:
        raise RuntimeError("SAM3 必须保持 eval()。")
    return model.forward_grounding(
        backbone_out=features,
        find_input=find_stage,
        find_target=None,
        geometric_prompt=make_geometric_prompt(boxes_cxcywh),
    )


def merge_topk_logits(outputs: dict, top_k: int) -> torch.Tensor:
    """按 presence 调整后的分类分数取 top-k，对 mask logits 做逐像素 max。"""
    if top_k < 1:
        raise ValueError(f"top_k 必须 >= 1，收到 {top_k}")
    pred_logits = outputs["pred_logits"]
    pred_masks = outputs["pred_masks"]
    if pred_logits.dim() != 3 or pred_masks.dim() != 4:
        raise RuntimeError(
            "Decoder 输出形状异常: "
            f"pred_logits {tuple(pred_logits.shape)}, pred_masks {tuple(pred_masks.shape)}"
        )

    scores = pred_logits.sigmoid()
    presence = outputs.get("presence_logit_dec")
    if presence is not None:
        scores = scores * presence.sigmoid().unsqueeze(1)
    query_scores = scores[0].squeeze(-1)
    query_masks = pred_masks[0]
    k = min(int(top_k), int(query_scores.numel()))
    index = torch.topk(query_scores, k=k).indices
    return query_masks.index_select(0, index).max(dim=0).values


def add_backbone_out(rgb_out, t_out):
    """与 demo10 相同，递归对两路 backbone 特征做 (RGB + T) / 2。"""
    if isinstance(rgb_out, torch.Tensor):
        return (rgb_out + t_out) / 2

    if isinstance(rgb_out, dict):
        return {key: add_backbone_out(rgb_out[key], t_out[key]) for key in rgb_out}

    if isinstance(rgb_out, list):
        return [add_backbone_out(rgb_value, t_value) for rgb_value, t_value in zip(rgb_out, t_out)]

    if rgb_out is None and t_out is None:
        return None


def encode_visual_text(model: nn.Module, device: torch.device) -> dict:
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        return model.backbone.forward_text(["visual"], device=device)


def forward_sample(
    rgb: torch.Tensor,
    thermal: torch.Tensor,
    gt_boxes: torch.Tensor,
    rgb_model: nn.Module,
    t_model: nn.Module,
    m1: M1 | None,
    text_outputs: dict,
    find_stage: FindStage,
    grad_backbone: bool = False,
) -> torch.Tensor:
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        if grad_backbone:
            rgb_out = rgb_model.backbone.forward_image(rgb)
            t_out = t_model.backbone.forward_image(thermal)
        else:
            with torch.no_grad():
                rgb_out = rgb_model.backbone.forward_image(rgb)
                t_out = t_model.backbone.forward_image(thermal)
        fused = m1(rgb_out, t_out) if m1 is not None else add_backbone_out(rgb_out, t_out)
        outputs = decode(rgb_model, branch_backbone(fused, text_outputs), gt_boxes, find_stage)
    return merge_topk_logits(outputs, int(gt_boxes.shape[0]))


def train_one_epoch(
    loader,
    rgb_model,
    t_model,
    m1,
    lora3,
    text_outputs,
    find_stage,
    optimizer,
    epoch: int,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    set_loader_epoch(loader, epoch)

    rgb_model.eval()
    t_model.eval()
    if m1 is not None:
        m1.train()
    if lora3 is not None:
        lora3.train()
        enable_trunk_checkpoint(rgb_model)
        enable_trunk_checkpoint(t_model)

    total_loss = 0.0
    total_step = len(loader)
    trainable = [param for group in optimizer.param_groups for param in group["params"]]
    for step, batch in enumerate(loader, start=1):
        optimizer.zero_grad(set_to_none=True)
        rgb = batch["rgb"].unsqueeze(0).to(device, non_blocking=True)
        thermal = batch["t"].unsqueeze(0).to(device, non_blocking=True)
        gt_boxes = batch["boxes"].to(device, non_blocking=True)
        gt_mask = batch["gt_mask"].to(device, non_blocking=True)

        pred_logit = forward_sample(
            rgb,
            thermal,
            gt_boxes,
            rgb_model,
            t_model,
            m1,
            text_outputs,
            find_stage,
            grad_backbone=lora3 is not None,
        )
        pred_map = upsample_logits(pred_logit, rgb.shape[-2:])
        gt = resize_gt(gt_mask, rgb.shape[-2:])

        str_term = args.str_weight * structure_loss(pred_map, gt)
        focal_term = args.focal_weight * sigmoid_focal_loss(pred_map, gt)
        dice_term = args.dice_weight * dice_loss(pred_map, gt)
        loss = str_term + focal_term + dice_term
        loss.backward()
        sync_gradients(trainable)
        optimizer.step()

        total_loss += float(loss.item())
        if step == 1 or step == total_step or step % args.log_every == 0:
            stats = torch.stack(
                [str_term.detach(), focal_term.detach(), dice_term.detach(), loss.detach()]
            ).float()
            if distributed():
                all_reduce_(stats, dist.ReduceOp.SUM)
                stats.div_(dist.get_world_size())
            str_value, focal_value, dice_value, loss_value = (float(value) for value in stats.tolist())
            message = (
                f"{datetime.now()} Epoch [{epoch:03d}/{args.epoch:03d}], "
                f"Step [{step:04d}/{total_step:04d}], "
                f"str: {str_value:.6f}, "
                f"focal: {focal_value:.6f}, "
                f"dice: {dice_value:.6f}, "
                f"total: {loss_value:.6f}"
            )
            if is_main_process():
                print(message)
            logging.info("#TRAIN# %s", message)

    totals = torch.tensor([total_loss, float(total_step)], dtype=torch.float64, device=device)
    if distributed():
        all_reduce_(totals, dist.ReduceOp.SUM)
    return float(totals[0].item() / max(totals[1].item(), 1.0))


@torch.no_grad()
def validate(loader, rgb_model, t_model, m1, lora3, text_outputs, find_stage, device: torch.device) -> dict:
    rgb_model.eval()
    t_model.eval()
    if m1 is not None:
        m1.eval()
    if lora3 is not None:
        lora3.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_mae = 0.0
    count = 0
    for batch in loader:
        rgb = batch["rgb"].unsqueeze(0).to(device, non_blocking=True)
        thermal = batch["t"].unsqueeze(0).to(device, non_blocking=True)
        gt_boxes = batch["boxes"].to(device, non_blocking=True)
        gt_mask = batch["gt_mask"]
        orig_h, orig_w = batch["orig_size"]
        if tuple(gt_mask.shape[-2:]) != (orig_h, orig_w):
            raise RuntimeError(
                f"{batch['stem']} 的 GT 尺寸 {tuple(gt_mask.shape[-2:])} 与 RGB 尺寸 {(orig_h, orig_w)} 不一致"
            )

        pred_logit = forward_sample(
            rgb, thermal, gt_boxes, rgb_model, t_model, m1, text_outputs, find_stage
        )
        probability = prediction_probability(pred_logit, gt_mask)
        tp, fp, fn = iou_counts(probability, gt_mask)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_mae += calc_mae(probability, gt_mask)
        count += 1

    stats = torch.tensor(
        [float(total_tp), float(total_fp), float(total_fn), total_mae, float(count)],
        dtype=torch.float64,
        device=device,
    )
    if distributed():
        all_reduce_(stats, dist.ReduceOp.SUM)
    tp, fp, fn, mae_sum, samples = stats.tolist()
    samples = int(samples)
    scale = max(samples, 1)
    return {"iou": dataset_iou(int(tp), int(fp), int(fn)), "mae": mae_sum / scale, "samples": samples}


def save_checkpoint(path: Path, m1: M1 | None, epoch: int, iou: float, lora3: nn.Module | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "m1": None if m1 is None else m1.state_dict(),
            "lora3": None if lora3 is None else lora3.state_dict(),
            "epoch": epoch,
            "iou": iou,
        },
        path,
    )


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="全监督 RGB-T 训练：冻结 SAM3 与双 LoRA，训练 M1 与共享 LoRA3",
        epilog="多卡示例: torchrun --nproc_per_node=4 train.py",
    )
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batchsize", type=int, default=1, help="必须为 1；多卡时每张卡各取 1 张图")
    parser.add_argument("--img_size", type=int, default=1008)
    parser.add_argument("--decay_rate", type=float, default=0.1)
    parser.add_argument("--decay_epoch", type=int, default=50)
    parser.add_argument("--str_weight", type=float, default=1.0)
    parser.add_argument("--focal_weight", type=float, default=1.0)
    parser.add_argument("--dice_weight", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=100)

    parser.add_argument("--lora", default=True, action=argparse.BooleanOptionalAction, help="关闭时 RGB、T、LoRA3 都不用，使用冻结的原始 SAM3")
    parser.add_argument("--lora3", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--m1", default=True, action=argparse.BooleanOptionalAction, help="关闭时用 (RGB+T)/2 代替 M1")
    parser.add_argument("--checkpoint", type=str, default="/data/ckpt_files/sam_ckpt/sam3.pt")
    parser.add_argument("--rgb_lora", type=str, default=str(ROOT / "outputs" / "RGB+LoRA" / "best" / "lora_weights.pt"))
    parser.add_argument("--t_lora", type=str, default=str(ROOT / "outputs" / "T+LoRA" / "best" / "lora_weights.pt"))
    parser.add_argument("--train_data_dir", type=str, default="/data/RGB-T-dataset/Seal/train")
    parser.add_argument("--val_data_dir", type=str, default="/data/RGB-T-dataset/Seal/val")
    parser.add_argument("--save_path", type=str, default="train_logs")
    return parser.parse_args()


def main() -> None:
    args = get_args()
    rank, world_size, device = init_distributed()
    try:
        if is_main_process():
            setup_logger(args.save_path, "train.log")
            logging.info("Args: %s", args)
            logging.info("GPU 数 %d，每卡 batch size %d", world_size, args.batchsize)
            print("Learning Rate:", args.lr)
            if world_size > 1:
                print(f"使用 {world_size} 张 GPU，每张卡每次 1 张图，梯度取平均。")
        else:
            logging.disable(logging.WARNING)

        checkpoint = Path(args.checkpoint)
        if args.lora:
            rgb_model = build_frozen_lora_model(checkpoint, Path(args.rgb_lora), device)
            t_model = build_frozen_lora_model(checkpoint, Path(args.t_lora), device)
            lora3 = build_shared_lora(rgb_model, t_model).to(device) if args.lora3 else None
        else:
            if args.lora3:
                logging.info("未加载模态 LoRA，LoRA3 一并关闭。")
            rgb_model = build_frozen_base_model(checkpoint, device)
            t_model = rgb_model
            lora3 = None
        m1 = M1().to(device) if args.m1 else None
        if distributed():
            broadcast_module(m1)
            broadcast_module(lora3)
            torch.manual_seed(rank)
            torch.cuda.manual_seed(rank)
        m1_trainable = 0 if m1 is None else count_trainable(m1)
        lora3_trainable = 0 if lora3 is None else count_trainable(lora3)
        rgb_trainable = count_trainable(rgb_model)
        t_trainable = count_trainable(t_model)
        logging.info(
            "可训练参数 m1=%d lora3=%d，冻结后 rgb=%d t=%d",
            m1_trainable,
            lora3_trainable,
            rgb_trainable,
            t_trainable,
        )
        if rgb_trainable or t_trainable:
            raise RuntimeError("SAM3 与模态 LoRA 应当全部冻结。")
        params = require_trainable(m1) if m1 is not None else []
        if lora3 is not None:
            lora3_params = [param for param in lora3.parameters() if param.requires_grad]
            if not lora3_params:
                raise RuntimeError("LoRA3 没有可训练参数。")
            params.extend(lora3_params)
        if not params:
            raise RuntimeError("M1 和 LoRA3 都已关闭，没有可训练参数。")

        multi_gpu = world_size > 1
        train_loader = get_loader(
            args.train_data_dir,
            image_size=args.img_size,
            batch_size=args.batchsize,
            shuffle=True,
            num_workers=args.num_workers,
            load_gt=True,
            distributed=multi_gpu,
            drop_last=multi_gpu,
        )
        val_loader = get_loader(
            args.val_data_dir,
            image_size=args.img_size,
            batch_size=args.batchsize,
            shuffle=False,
            num_workers=args.num_workers,
            load_gt=True,
            distributed=multi_gpu,
        )
        require_equal_steps(train_loader, device)
        if multi_gpu and is_main_process():
            used = len(train_loader) * world_size
            total = len(train_loader.dataset)
            logging.info("训练集 %d 张，每轮使用 %d 张。", total, used)
            if used != total:
                print(f"训练集 {total} 张不能被 {world_size} 张卡整除，每轮使用 {used} 张。")
        text_outputs = encode_visual_text(rgb_model, device)
        find_stage = make_find_stage(device)
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

        best_iou = -1.0
        best_epoch = 0
        checkpoint_path = Path(args.save_path) / "best.pth"
        if is_main_process():
            print("Go! Go! Go!")
        for epoch in range(1, args.epoch + 1):
            lr = adjust_lr(optimizer, args.lr, epoch, decay_rate=args.decay_rate, decay_epoch=args.decay_epoch)
            logging.info("Epoch %d lr %.8f", epoch, lr)
            train_loss = train_one_epoch(
                train_loader,
                rgb_model,
                t_model,
                m1,
                lora3,
                text_outputs,
                find_stage,
                optimizer,
                epoch,
                args,
                device,
            )
            logging.info("#TRAIN# epoch=%d mean_loss=%.6f", epoch, train_loss)

            metrics = validate(val_loader, rgb_model, t_model, m1, lora3, text_outputs, find_stage, device)
            if metrics["samples"] > 0 and metrics["iou"] > best_iou:
                best_iou = metrics["iou"]
                best_epoch = epoch
                if is_main_process():
                    save_checkpoint(checkpoint_path, m1, epoch, metrics["iou"], lora3)
                    logging.info("保存最佳权重 %s", checkpoint_path)
            if distributed():
                dist.barrier()

            message = (
                f"Epoch: {epoch} IoU: {format_metric(metrics['iou'])} MAE: {metrics['mae']:.6f} "
                f"bestIoU: {format_metric(best_iou)} bestEpoch: {best_epoch} samples: {metrics['samples']}"
            )
            if is_main_process():
                print(message)
            logging.info("#VAL# %s", message)

        logging.info("训练结束。最佳验证 IoU %s，出现在 epoch %d", format_metric(best_iou), best_epoch)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
