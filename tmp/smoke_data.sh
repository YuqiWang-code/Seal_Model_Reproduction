#!/bin/bash
set -e
VENV=/share_datasets/Seal_Model_Reproduction/Ours/venv_ours
PY=$VENV/bin/python
SEAL=/share_datasets/Seal_Model_Reproduction/Seal_data
ROOT=/share_datasets/Seal_Model_Reproduction/Ours

echo "=== original loader ==="
cd $ROOT/Original_Code
$PY - <<PY
import torch
from data import resolve_seal_split_paths, get_loader, TestDataset
p = resolve_seal_split_paths("$SEAL/train")
print("paths:", {k: str(v) for k, v in p.items()})
loader = get_loader(image_root=p["rgb"], thermal_root=p["t"], bbox_root=p["bbox"],
                    batchsize=1, trainsize=1008, shuffle=False, num_workers=0,
                    train_mode=True, require_names=True)
pack, name = next(iter(loader))
rgb, t, boxes, hw = pack
print("name", name, "rgb", tuple(rgb.shape), rgb.dtype, "t", tuple(t.shape),
      "boxes", boxes.shape, "hw", hw.tolist())
print("rgb range", float(rgb.min()), float(rgb.max()), "t range", float(t.min()), float(t.max()))
td = TestDataset(image_root="$SEAL/val/RGB", gt_root="$SEAL/val/GT", thermal_root="$SEAL/val/T",
                 testsize=1008, sam3_image_size=1008)
r, gt, tt, nm, _ = td.load_data()
print("test sample", nm, tuple(r.shape), tuple(tt.shape), gt.size, gt.mode)
PY

echo "=== fully-supervised loader ==="
cd $ROOT/Fully_Supervised_Code_1104
$PY - <<PY
import torch
from data import get_loader
loader = get_loader("$SEAL/train", image_size=1008, batch_size=1, shuffle=False,
                    num_workers=0, load_gt=True)
batch = next(iter(loader))
print("stem", batch["stem"], "rgb", tuple(batch["rgb"].shape), "t", tuple(batch["t"].shape),
      "boxes", batch["boxes"].shape, "gt", tuple(batch["gt_mask"].shape), batch["gt_mask"].dtype)
print("rgb range", float(batch["rgb"].min()), float(batch["rgb"].max()),
      "t range", float(batch["t"].min()), float(batch["t"].max()))
PY
echo "data smoke OK"
