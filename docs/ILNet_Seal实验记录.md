# ILNet：Seal 全监督训练与测试记录

## 方法与来源

| 项目 | 值 |
|---|---|
| 方法 | ILNet（Low-Level Matters for Salient Infrared Small Target Detection） |
| 论文 | Low-Level Matters for Salient Infrared Small Target Detection，IEEE TAES 2025 |
| 代码 | <https://github.com/Li-Haoqing/ILNet>（作者官方实现） |
| 目录 | `ILNet-main/`（ILNet-L 配置） |
| 输入 | Seal 热红外 `T`（3 通道） |
| 监督 | 全像素 GT（Seal train 1,500 张） |

来源：awesome-infrared-small-targets 的 Single-Frame 条目（2025）。方法强调低层特征与侧输出监督，模型很小（ILNet-S/M/L 三档）。

## 训练协议

| 项目 | 值 |
|---|---|
| 网络 | `ILNet_L(out_ch=1)`，官方 L 档结构 |
| 损失 | 官方 `criterion`：所有侧输出（含主输出）的 BCE-with-logits 之和 |
| 优化器 | 官方 Adan，lr 1e-3，weight decay 1e-4，5 epoch warm-up + cosine（`create_lr_scheduler`） |
| epoch / batch | 300 / 8（官方 600 / 8；按项目预算减半） |
| 裁剪 | 256×256（官方把整图 resize 到 512；这里保留原生尺度的 256 裁剪） |
| 模型选择 | Seal val 滑窗像素 IoU@0.5 最优；test 不参与 |
| 测试推理 | 原生分辨率 256 滑窗、步长 128 |

## 运行命令

```bash
cd /share_datasets/Seal_Model_Reproduction/ILNet-main
PY=/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python
$PY train_seal.py --device cuda:0 --epochs 300 --val-interval 20 --batch-size 8 --out experiments/Seal_fullsup
```

## 结果

选中的是 Seal val 上像素 IoU 最高的第 220 epoch 检查点（val IoU 0.25066；训练跑满 300 epoch）。测试集 1,983 张预测全部生成。

| 来源 | S↑ | Fw↑ | E↑ | MAE↓ |
|---|---:|---:|---:|---:|
| ILNet，val 最优检查点，test 1,983 张 | 0.49894 | 0.31966 | 0.68460 | 0.00030902 |

| IoU↑ | F1↑ | Pd↑ | Fa↓（原始比例） |
|---:|---:|---:|---:|
| 0.19172233 | 0.32175672 | 0.58946228 | 1.9302457931986373e-05 |

四项指标由根目录 `evaluate.py` 的四个原函数计算，小目标四项由 `evaluate_small_targets.py` 在同一批预测上计算。完整精度见 `docs/ILNet_Seal评估产物/eight_metrics.json`。

## 产物与已知问题

- 权重：`ILNet-main/experiments/Seal_fullsup/{best.pth,last.pth}`；日志 `logs/ilnet_train.log`。
- 预测：`results/ILNet/Seal/fullsup_test_maps/`。
- 官方 `utils/data.py` 用 SIRST 的均值/方差归一化；本实验统一用 ImageNet 均值方差，与其他全监督基线一致。
- 与官方配置的差异：epoch 600→300、输入改为原生 256 裁剪（官方 resize 512）、验证改用 Seal val。
