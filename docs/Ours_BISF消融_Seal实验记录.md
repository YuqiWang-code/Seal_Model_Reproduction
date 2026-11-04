# Ours B+BISF（M1/BISF 消融）：Seal 训练与测试记录

对应 `docs/Seal量化SOTA表.xlsx` 的 **Ablation** 表第 15 行 `B+BISF`；对照基线是第 14 行 `B`
（`(RGB + T) / 2` 融合，数值 S 0.49897 / Fw 0.63763 / E 0.67956 / MAE 0.00023 /
IoU 0.4216 / F1 0.59314 / Pd 0.8143 / Fa 5.57e-05）。

## 代码与协议

代码由同门提供的最新消融包（原 `Ours/m1_ablation/`）替换进 `Ours/Fully_Supervised_Code_1104/`，
`m1_ablation/` 已删除。该包只跑一种设定：**不加载 RGB LoRA、T LoRA 与 LoRA3，RGB 与 T 共用
一份冻结的原始 SAM3，只训练 M1（即 BISF 模块）**；训练时 M1 融合两路视觉特征，用 `train/box`
的真实框作 Decoder 的几何提示，预测 mask 用 `train/GT` 做结构损失 + Sigmoid Focal + Dice。

| 项目 | 值 |
|---|---|
| 输入 | Seal `train/RGB`（jpg）+ `train/T`（16 位 TIFF，逐图 min-max → 8 位 → RGB），1008×1008 |
| 冻结 | SAM3 全部参数（bf16） |
| 可训练 | 仅 `M1`（BISF）：1.19M 参数、4.77 MB 的 `best.pth` |
| 监督 | `train/box` 作几何提示 + `train/GT` 全像素 mask；结构损失（加权 BCE+加权 IoU）+ Focal(α=0.25, γ=2) + Dice |
| 优化器 | AdamW，lr 5e-5，weight_decay 0.01，30 epoch，batch 1 |
| 选点 | Seal **val** 500 张的像素 IoU@0.5 最优（`prob > 0.5`、GT `>128`、全数据集累计）；test 不参与 |
| 测试 | `test/RGB`+`test/T`，同一入口（真实框提示）→ sigmoid → 恢复 GT 尺寸 → `prob×255` 的 8 位 PNG |
| 指标 | 根目录 `evaluate.py`（四原函数）+ `evaluate_small_targets.py`（IoU/F1/Pd/Fa）；包内 `evaluate3.py` 为同口径交叉核对 |

## 相对同门包的适配（只动运行与输入）

- `data.py` 的 `load_t_image` 恢复逐图 min-max：原式 `x - min/(max-min)*255` 只适用 8 位图，
  Seal 的 `T` 实测是 uint16（26,400–28,500），原式转 uint8 会回绕成噪声。
- 设备改由 `SEAL_CUDA` 控制（默认 GPU1）；`--checkpoint` 指向 `Ours/weights/sam3.pt`；
  数据指向 `Seal_data/{train,val,test}`；`--save_path` 默认 `results/Ours/BISF`。
- `test.py`、`evaluate3.py` 的默认路径同步改为本服务器；`evaluate3.py` 增加 `--json_output`。
- 旧的全监督 LoRA 变体（Quantity 表 34/35 行）代码归档到 `Ours/archive/FullySup_LoRA_1104/`。

## 运行

```bash
bash /share_datasets/Seal_Model_Reproduction/Ours/Fully_Supervised_Code_1104/run_train.sh
bash /share_datasets/Seal_Model_Reproduction/Ours/Fully_Supervised_Code_1104/run_test.sh
```

冒烟（迷你 24/6 子集 1 epoch + 6 张推理 + evaluate3）已跑通。

## 结果

<!--RESULTS-->

## 产物

- 权重：`results/Ours/BISF/best.pth`（仅 M1）；日志 `logs/ours_bisf_train.log`
- 预测：`results/Ours/BISF/test_maps/`（1,983 张）
- 指标：`results/Ours/BISF/metrics4.json`、`metrics_small.json`（根目录统一入口）；
  `evaluate3.json`（包内同口径交叉核对）
