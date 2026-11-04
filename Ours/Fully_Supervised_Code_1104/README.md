# Fully_Supervised_Code_1104（现为 M1/BISF 消融版）

由同门提供的最新消融代码（原 `m1_ablation/`）替换而来，只跑一种设定：
**不用 RGB LoRA、T LoRA 和 LoRA3，两路图像共用一份冻结的原始 SAM3，只训练 M1**。
这里的 M1 就是 BISF 模块；对应 `docs/Seal量化SOTA表.xlsx` 的 Ablation 表 `B+BISF` 行。

## 目录约定

```
<split>/RGB/*.jpg
<split>/T/*.tif        # Seal 是 16 位 TIFF
<split>/box/*.txt      # 绝对像素 x1 y1 x2 y2，作为 Decoder 的几何提示
<split>/GT/*.png       # train/val/test 都要有
```

## 相对同门原始包的改动（只动运行与输入适配）

| 文件 | 改动 |
|---|---|
| `data.py` | `load_t_image` 改回逐图 min-max：原式 `x - min/(max-min)*255` 只适用 8 位图，Seal 的 T 是 uint16（实测 26,400–28,500），原式转 uint8 会回绕成噪声 |
| `train.py` | 设备由 `SEAL_CUDA` 控制（默认 1）；`--checkpoint`、`--train_data_dir`、`--val_data_dir`、`--save_path` 默认值指向本服务器（`Ours/weights/sam3.pt`、`Seal_data/*`、`results/Ours/BISF`）；新增可选 `--seed` |
| `test.py` | 设备同上；`--checkpoint`、`--sam_checkpoint`、`--test_data_dir`、`--save_dir` 默认值指向本服务器 |
| `evaluate3.py` | 默认 GT 改到 `Seal_data/test/GT`，新增可选 `--json_output` |
| `run_train.sh` / `run_test.sh` | 改成用 `Ours/venv_ours`、本服务器路径，直接可跑 |

`SAM3_LoRA/lora_layers.py` 与 `SAM3_LoRA/sam3/` 保持同门版本（本消融不使用 LoRA，`--lora` 时才加载）。
`evaluate.py`、`metrics.py` 与项目根目录的四个原函数、`AS1000Metrics` 逐字节一致，口径不变。
旧的全监督 LoRA 变体（对应 Quantity 表 34/35 行）已归档到 `Ours/archive/FullySup_LoRA_1104/`。

## 训练（B+BISF）

```bash
bash Ours/Fully_Supervised_Code_1104/run_train.sh          # 30 epoch，val IoU 选点
```

权重只含 M1（`best.pth` 的 `m1` 字段），日志 `train.log`。

## 测试与指标

```bash
bash Ours/Fully_Supervised_Code_1104/run_test.sh
```

预测写到 `results/Ours/BISF/test_maps/`（1,983 张，概率×255 的 8 位 PNG，文件名同 GT），
指标由根目录 `evaluate.py`（四原函数）与 `evaluate_small_targets.py`（IoU/F1/Pd/Fa）计算，
JSON 存 `metrics4.json` / `metrics_small.json`；包内 `evaluate3.py` 可作独立交叉核对（同一口径）。

## 基线 B（对照，不填表）

同一脚本加 `--no-m1` 即 `(RGB+T)/2` 融合的 B 设定：

```bash
python train.py --no-m1 --save_path /share_datasets/Seal_Model_Reproduction/results/Ours/B
```
