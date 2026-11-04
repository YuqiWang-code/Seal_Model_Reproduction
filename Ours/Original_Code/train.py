import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("SEAL_CUDA", "1"))

import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import TestDataset, get_loader, resolve_seal_split_paths
from sam3.sam3model import Model
from utils import adjust_lr, setup_logger, LocalSaliencyCoherence, projection_loss


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SSFam-style SAM3 training")
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batchsize", type=int, default=1)
    parser.add_argument("--img_size", type=int, default=1008)
    parser.add_argument("--decay_rate", type=float, default=0.1)
    parser.add_argument("--decay_epoch", type=int, default=50)
    parser.add_argument("--fg_loss_weight", type=float, default=0.05)
    parser.add_argument("--fg_min_prob", type=float, default=0.15)

    parser.add_argument("--checkpoint", type=str, default="/share_datasets/Seal_Model_Reproduction/Ours/weights/sam3.pt")
    parser.add_argument("--resnet_ckpt", type=str, default="/share_datasets/Seal_Model_Reproduction/Ours/weights/resnet50-19c8e357.pth")
    parser.add_argument("--save_path", type=str, default="train_logs")
    parser.add_argument("--save_name", type=str, default="best3.pth")
    parser.add_argument("--t_encoder_type", type=str, default="resnet50", choices=["resnet50", "lora"])

    parser.add_argument("--train_data_dir", type=str, default="/share_datasets/Seal_Model_Reproduction/Seal_data/train")
    parser.add_argument("--val_data_dir", type=str, default="/share_datasets/Seal_Model_Reproduction/Seal_data/val")
    parser.add_argument("--k_folds", type=int, default=1)
    return parser.parse_args()


def structure_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred_prob = torch.sigmoid(pred)
    inter = ((pred_prob * mask) * weit).sum(dim=(2, 3))
    union = ((pred_prob + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def train_one_epoch(
    loader,
    model: Model,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    lsc_loss: LocalSaliencyCoherence,
) -> None:
    model.train()
    total_step = len(loader)

    lsc_kernels = [{"weight": 1, "xy": 6, "rgb": 0.1}]
    lsc_radius = 5

    for i, pack in enumerate(loader, start=1):
        optimizer.zero_grad(set_to_none=True)

        if isinstance(pack, (tuple, list)) and len(pack) == 2:
            raw_data, _sample_name = pack
        else:
            raw_data = pack

        rgb, t_img, box_prompts, _out_hw = raw_data

        if rgb.dim() == 3:
            rgb = rgb.unsqueeze(0)
        if t_img.dim() == 3:
            t_img = t_img.unsqueeze(0)

        rgb = rgb.cuda(non_blocking=True)
        t_img = t_img.cuda(non_blocking=True)

        if not isinstance(box_prompts, torch.Tensor):
            raise RuntimeError("Box prompts are required, but got None.")
        box_prompts = box_prompts.cuda(non_blocking=True)
        if box_prompts.dim() == 1:
            box_prompts = box_prompts.unsqueeze(0)
        if box_prompts.dim() == 3 and box_prompts.shape[0] == 1:
            box_prompts = box_prompts.squeeze(0)

        all_boxes = box_prompts.view(-1, 4)

        image_embed, high_res_feats = model.encode_images(rgb, t_img)

        mask_1_dec, _noprompt_feat_dec = model.decode_noprompt(
            image_embed,
            high_res_feats,
        )

        mask_1_dec_up = F.interpolate(mask_1_dec, size=rgb.shape[-2:], mode="bilinear", align_corners=False)

        prompt_logits = []
        for one_box in all_boxes:
            mask_2_i, _prompt_feat = model.decode_prompt(
                image_embed,
                high_res_feats,
                box_prompts=one_box.view(1, 4),
                mask_prompts=None,
            )
            prompt_logits.append(mask_2_i)

        mask_2 = torch.stack(prompt_logits, dim=0).mean(dim=0)
        mask_2_up = F.interpolate(mask_2, size=rgb.shape[-2:], mode="bilinear", align_corners=False)

        # mask1 losses (allow str loss to backprop through encoder path)
        should_log = (i % 100 == 0 or i == total_step or i == 1)

        str_loss = structure_loss(mask_1_dec_up, torch.sigmoid(mask_2_up.detach()))

        # mask2 losses
        mask_prob_2 = torch.sigmoid(mask_2_up)
        proj_loss_2 = projection_loss(mask_2_up, all_boxes)

        B, _, H, W = mask_prob_2.shape
        x_coords = torch.arange(W, device=mask_prob_2.device).view(1, 1, W)
        y_coords = torch.arange(H, device=mask_prob_2.device).view(1, 1, H)
        boxes = all_boxes.view(1, -1, 4)
        x1, y1, x2, y2 = boxes[..., 0:1], boxes[..., 1:2], boxes[..., 2:3], boxes[..., 3:4]
        in_box_x = (x_coords >= x1) & (x_coords <= x2)
        in_box_y = (y_coords >= y1) & (y_coords <= y2)
        box_mask = (in_box_y.unsqueeze(-1) & in_box_x.unsqueeze(-2)).float().squeeze(2)
        in_box_mean = (mask_prob_2.squeeze(1).unsqueeze(1) * box_mask).sum(dim=(2, 3)) / (box_mask.sum(dim=(2, 3)) + 1e-6)
        fg_loss_2 = args.fg_loss_weight * F.relu(args.fg_min_prob - in_box_mean.max(dim=1).values).mean()

        rgb_quarter = F.interpolate(rgb, scale_factor=0.25, mode="bilinear", align_corners=True)
        sample_rgb = {"rgb": rgb_quarter}
        result_quarter = F.interpolate(mask_prob_2, scale_factor=0.25, mode="bilinear", align_corners=True)
        lsc_loss_2 = lsc_loss(
            result_quarter,
            lsc_kernels,
            lsc_radius,
            sample_rgb,
            rgb_quarter.shape[2],
            rgb_quarter.shape[3],
        )["loss"]

        loss = (str_loss + proj_loss_2 + lsc_loss_2 + fg_loss_2).mean()

        loss.backward()
        optimizer.step()

        if should_log:
            msg = (
                f"{datetime.now()} Epoch [{epoch:03d}/{args.epoch:03d}], "
                f"Step [{i:04d}/{total_step:04d}], "
                f"str_loss: {str_loss.item():.6f}, "
                f"proj_loss2: {proj_loss_2.item():.6f}, "
                f"lsc_loss2: {lsc_loss_2.item():.6f}, "
                f"fg_loss2: {fg_loss_2.item():.6f}, "
                f"total_loss: {loss.item():.6f}"
            )
            print(msg)
            logging.info("#TRAIN# " + msg)


def validate(
    loader: TestDataset,
    model: Model,
    epoch: int,
    best_mae: float,
    best_epoch: int,
    args: argparse.Namespace,
):
    model.eval()
    mae_sum = 0.0

    with torch.no_grad():
        for _ in range(len(loader)):
            rgb, gt_pil, t_img, _name, _out_hw = loader.load_data()
            gt = np.asarray(gt_pil, np.float32)
            gt = gt / (gt.max() + 1e-8)

            rgb = rgb.cuda(non_blocking=True)
            t_img = t_img.cuda(non_blocking=True)

            pred, _, _, _ = model(rgb, t_img, None)
            pred = F.interpolate(pred, size=gt.shape, mode="bilinear", align_corners=False)
            pred = pred.sigmoid().data.cpu().numpy().squeeze()
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

            mae_sum += np.sum(np.abs(pred - gt)) / (gt.shape[0] * gt.shape[1])

    mae = mae_sum / len(loader)

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    is_best = mae < best_mae
    if is_best:
        best_mae = mae
        best_epoch = epoch
        torch.save(model.state_dict(), str(save_path / args.save_name))

    print(f"Epoch: {epoch} MAE: {mae:.6f} bestMAE: {best_mae:.6f} bestEpoch: {best_epoch}")
    logging.info(
        f"#TEST# Epoch:{epoch} MAE:{mae:.6f} bestEpoch:{best_epoch} bestMAE:{best_mae:.6f}"
    )
    return best_mae, best_epoch


def main() -> None:
    args = get_args()
    setup_logger(args.save_path, "log3.log")

    print("Learning Rate:", args.lr)
    logging.info(f"Args: {args}")

    model = Model(
        ckpt_path=args.checkpoint,
        image_size=args.img_size,
        resnet_ckpt=args.resnet_ckpt,
    ).cuda()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.lr)

    train_paths = resolve_seal_split_paths(args.train_data_dir)
    val_paths = resolve_seal_split_paths(args.val_data_dir)

    train_loader = get_loader(
        image_root=train_paths["rgb"],
        thermal_root=train_paths["t"],
        bbox_root=train_paths.get("bbox", train_paths["rgb"]),
        batchsize=args.batchsize,
        trainsize=args.img_size,
        k_folds=args.k_folds,
        train_mode=True,
        sam3_image_size=args.img_size,
        require_names=True,
    )

    val_loader = TestDataset(
        image_root=val_paths["rgb"],
        gt_root=val_paths["gt"],
        thermal_root=val_paths["t"],
        testsize=args.img_size,
        sam3_image_size=args.img_size,
    )

    lsc_loss = LocalSaliencyCoherence().cuda()

    best_mae = float("inf")
    best_epoch = 0

    print("Starting!")
    for epoch in range(1, args.epoch + 1):
        adjust_lr(optimizer, args.lr, epoch, decay_rate=args.decay_rate, decay_epoch=args.decay_epoch)
        train_one_epoch(train_loader, model, optimizer, epoch, args, lsc_loss)
        best_mae, best_epoch = validate(val_loader, model, epoch, best_mae, best_epoch, args)


if __name__ == "__main__":
    main()
