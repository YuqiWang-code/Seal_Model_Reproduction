"""原始版训练步基准（随机初始化，不需要权重）：3 步，报告耗时/显存。

复刻 train.py 的 train_one_epoch：encode_images -> 无提示解码 / 框提示解码 ->
结构损失(mask1 vs mask2) + 投影损失 + LSC + 框内前景损失。
默认 fp32（与原 train.py 一致）；--amp 时用 bf16 自动混合精度。
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path("/share_datasets/Seal_Model_Reproduction")
sys.path.insert(0, str(ROOT / "Ours" / "Original_Code"))

from data import get_loader, resolve_seal_split_paths  # noqa: E402
from sam3.sam3model import Model  # noqa: E402
from utils import LocalSaliencyCoherence, projection_loss  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--amp", action="store_true")
parser.add_argument("--steps", type=int, default=3)
args = parser.parse_args()

SEAL = ROOT / "Seal_data"
DEVICE = torch.device("cuda")
IMG = 1008
FG_W, FG_MIN = 0.05, 0.15


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred_prob = torch.sigmoid(pred)
    inter = ((pred_prob * mask) * weit).sum(dim=(2, 3))
    union = ((pred_prob + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


print("building original Model (random init) ...", flush=True)
t0 = time.time()
model = Model(ckpt_path=None, image_size=IMG, t_encoder_type="resnet50",
              resnet_ckpt=str(ROOT / "Ours" / "weights" / "resnet50-19c8e357.pth")).cuda()
print(f"built in {time.time() - t0:.1f}s, trainable(M) "
      f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}", flush=True)
model.train()
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], 5e-5)
lsc = LocalSaliencyCoherence().cuda()

paths = resolve_seal_split_paths(str(SEAL / "train"))
loader = get_loader(image_root=paths["rgb"], thermal_root=paths["t"], bbox_root=paths["bbox"],
                    batchsize=1, trainsize=IMG, shuffle=False, num_workers=2,
                    train_mode=True, require_names=True)

torch.cuda.reset_peak_memory_stats()
step = 0
for pack in loader:
    step += 1
    rgb, t_img, box_prompts, _ = pack[0]
    rgb = rgb.unsqueeze(0).cuda()
    t_img = t_img.unsqueeze(0).cuda()
    all_boxes = box_prompts.view(-1, 4).cuda()
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if args.amp else torch.autocast("cuda", enabled=False)
    with ctx:
        image_embed, high_res = model.encode_images(rgb, t_img)
        mask_1, _ = model.decode_noprompt(image_embed, high_res)
        mask_1_up = F.interpolate(mask_1, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        prompt_logits = [model.decode_prompt(image_embed, high_res, box_prompts=b.view(1, 4))[0]
                         for b in all_boxes]
        mask_2 = torch.stack(prompt_logits, dim=0).mean(dim=0)
        mask_2_up = F.interpolate(mask_2, size=rgb.shape[-2:], mode="bilinear", align_corners=False)

        str_loss = structure_loss(mask_1_up, torch.sigmoid(mask_2_up.detach()))
        proj = projection_loss(mask_2_up, all_boxes)

        mask_prob_2 = torch.sigmoid(mask_2_up)
        B, _, H, W = mask_prob_2.shape
        xc = torch.arange(W, device=mask_prob_2.device).view(1, 1, W)
        yc = torch.arange(H, device=mask_prob_2.device).view(1, 1, H)
        boxes = all_boxes.view(1, -1, 4)
        x1, y1, x2, y2 = boxes[..., 0:1], boxes[..., 1:2], boxes[..., 2:3], boxes[..., 3:4]
        in_x = (xc >= x1) & (xc <= x2)
        in_y = (yc >= y1) & (yc <= y2)
        box_mask = (in_y.unsqueeze(-1) & in_x.unsqueeze(-2)).float().squeeze(2)
        in_mean = (mask_prob_2.squeeze(1).unsqueeze(1) * box_mask).sum(dim=(2, 3)) / (box_mask.sum(dim=(2, 3)) + 1e-6)
        fg = FG_W * F.relu(FG_MIN - in_mean.max(dim=1).values).mean()

        q = F.interpolate(rgb, scale_factor=0.25, mode="bilinear", align_corners=True)
        rq = F.interpolate(mask_prob_2, scale_factor=0.25, mode="bilinear", align_corners=True)
        lsc_l = lsc(rq, [{"weight": 1, "xy": 6, "rgb": 0.1}], 5, {"rgb": q}, q.shape[2], q.shape[3])["loss"]

        loss = (str_loss + proj + lsc_l + fg).mean()

    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    print(f"step {step}: {time.time() - t0:.2f}s str={float(str_loss):.5f} proj={float(proj):.5f} "
          f"lsc={float(lsc_l):.5f} fg={float(fg):.5f} peak={torch.cuda.max_memory_allocated() / 2**30:.1f}GiB",
          flush=True)
    if step >= args.steps:
        break
print("original train-step smoke OK (amp=%s)" % args.amp)
