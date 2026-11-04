# Original_Code：原始 Ours（RGB-T + 框提示的 SAM3 双解码器）

作者原始代码，原样保留（只改了运行所需的路径/设备/依赖）。协议：SAM3 tracker 主干，
RGB 走 LoRA、T 走 ResNet50 温度分支，融合后由两个 mask decoder 分别做无提示解码
（`mask_1`）与真实框提示解码（`mask_2`）；训练用 `mask_2` 监督 `mask_1`
（结构损失），再加投影损失、LSC 平滑与框内前景损失。测试只用 `mask_1`（无提示分支）。

## 相对原始代码的改动（只动运行相关，不动损失与网络）

| 文件 | 改动 |
|---|---|
| `utils.py` | 新增 `open_thermal`：Seal 的 16 位 `T` 逐图 min-max → 8 位 → RGB（原 `open_rgb` 会把 16 位截断到 255） |
| `data.py` | 箱目录同时接受 Seal 的 `box/`（原来只认 `bbox/`）；T 分支改用 `open_thermal` |
| `sam3/sam3model.py` | `Model(..., resnet_ckpt=...)`：ResNet50 权重路径可配（原为硬编码 `/data/...`），缺失时给明确报错 |
| `train.py` / `test.py` | 设备由 `SEAL_CUDA`（默认 1）控制（原硬编码 '3'）；数据与权重默认路径改到本项目服务器路径；`train.py` 增加 `--resnet_ckpt` |

## 服务器运行

```bash
ROOT=/share_datasets/Seal_Model_Reproduction
PY=$ROOT/Ours/venv_ours/bin/python    # 独立 venv，补了 iopath/hydra/decord 等 SAM3 依赖
cd $ROOT/Ours/Original_Code

# 训练：Seal train 训练，Seal val 选模型（MAE 最优），test 不参与
SEAL_CUDA=1 $PY train.py --epoch 30 --lr 5e-5 --img_size 1008 \
  --checkpoint $ROOT/Ours/weights/sam3.pt \
  --resnet_ckpt $ROOT/Ours/weights/resnet50-19c8e357.pth \
  --train_data_dir $ROOT/Seal_data/train --val_data_dir $ROOT/Seal_data/val \
  --save_path train_logs --save_name best3.pth 2>&1 | tee train_original.log

# 测试：Seal test 1,983 张 → 8 位灰度 PNG
SEAL_CUDA=1 $PY test.py --mode single --checkpoint train_logs/best3.pth \
  --test_data_dir $ROOT/Seal_data/test \
  --save_dir $ROOT/results/Ours/Original/test_maps

# 八项指标（项目根目录统一入口）
cd $ROOT
$PY evaluate.py --pred_root $ROOT/results/Ours/Original/test_maps \
  --gt_dir $ROOT/Seal_data/test/GT --json_output $ROOT/results/Ours/Original/metrics4.json
$PY evaluate_small_targets.py --pred_root $ROOT/results/Ours/Original/test_maps \
  --gt_dir $ROOT/Seal_data/test/GT --json_output $ROOT/results/Ours/Original/metrics_small.json
```

`test.py` 默认把结果写到 `--save_dir/<checkpoint 名>/test/`，上例按需再挪一层。
