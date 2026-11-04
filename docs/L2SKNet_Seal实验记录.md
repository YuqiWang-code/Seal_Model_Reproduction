# L2SKNet：Seal 全监督训练与测试记录

## 方法与来源

| 项目 | 值 |
|---|---|
| 方法 | L2SKNet（Saliency at the Helm: Steering Infrared Small Target Detection with Learnable Kernels） |
| 论文 | TGRS 2025，GrokCV 组工作 |
| 代码 | <https://github.com/fengyiwu98/L2SKNet>（作者官方实现） |
| 目录 | `L2SKNet-main/` |
| 输入 | Seal 热红外 `T`（单通道） |
| 监督 | 全像素 GT（Seal train 1,500 张） |

来源：GrokCV 出版物页（TGRS 2025）+ awesome-infrared-small-targets 的 Single-Frame 条目。

## 训练协议

| 项目 | 值 |
|---|---|
| 网络 | `Net('L2SKNet_UNet')`，官方默认 U-Net 变体 |
| 损失 | dataset 级 SoftIoU（见下） |
| 优化器 | Adam，lr 5e-4，MultiStepLR [100,140] ×0.1（官方 train_device0.py 设置） |
| epoch / batch | 150 / 8（官方 400 / 8） |
| 裁剪 | 256×256，随机裁剪 + 翻转 |
| 模型选择 | Seal val 滑窗像素 IoU@0.5 最优；test 不参与 |
| 测试推理 | 原生分辨率 256 滑窗、步长 128 |

## 运行命令

```bash
cd /share_datasets/Seal_Model_Reproduction/L2SKNet-main
PY=/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python
$PY train_seal.py --device cuda:0 --epochs 150 --val-interval 10 --batch-size 8 --out experiments/Seal_fullsup
```

## 结果

选中的是 Seal val 上像素 IoU 最高的第 130 epoch 检查点（val IoU 0.34082；训练跑满 150 epoch）。

| 来源 | S↑ | Fw↑ | E↑ | MAE↓ |
|---|---:|---:|---:|---:|
| L2SKNet，val 最优检查点，test 1,983 张 | 0.49811 | 0.52043 | 0.67773 | 0.00028301 |

| IoU↑ | F1↑ | Pd↑ | Fa↓（原始比例） |
|---:|---:|---:|---:|
| 0.30092370 | 0.46263082 | 0.77204388 | 8.624084674547941e-05 |

四项指标由根目录 `evaluate.py` 的四个原函数计算，小目标四项由 `evaluate_small_targets.py` 在同一批预测上计算。完整精度见 `docs/L2SKNet_Seal评估产物/eight_metrics.json`。

## 复杂度

| Params↓ (M) | GFLOPs↓ | FPS↑ | 单次前向 (ms) |
|---:|---:|---:|---:|
| 0.899 | 6.895 | 332.16 | 3.011 |

单窗 256×256、FP32、batch 1、RTX 5090 实测（thop MACs；三轮 × 每轮 100 次计时，取最好一轮的中位数）。Seal test 一张 622×417 需 12 个滑窗，单图成本约 ×12、单图吞吐约 FPS÷12。对照：本仓库 `t_models.py`（thop、256²）打印 0.898913 M / 6.894535 G，与实测逐位一致；DRPCA-Net 对比表亦为 0.899 M。LPCNet 表的 1.07 M / 5.92 G 应来自另一份实现，不作基准。

## 产物与已知问题

- 权重：`L2SKNet-main/experiments/Seal_fullsup/{best.pth,last.pth}`；日志 `logs/l2sknet_maintrain.log`。
- 预测：`results/L2SKNet/Seal/fullsup_test_maps/`。
- **一处训练改动**：官方逐样本 SoftIoU 在 Seal 上 20 epoch 内 val IoU 恒为 0.0001（训练 loss 仍在降），按本项目对 RPCANet 系列的同一处理改成 dataset 级 SoftIoU；网络、优化器、学习率日程与 epoch 不变。
- 与官方配置的差异：epoch 400→150、验证改用 Seal val。
