# RPCANet++：Seal 全监督训练与测试记录

## 方法与来源

| 项目 | 值 |
|---|---|
| 方法 | RPCANet++（Deep Interpretable Robust PCA for Sparse Object Segmentation） |
| 论文 | RPCANet++: Deep Interpretable Robust PCA for Sparse Object Segmentation，arXiv 2508.04190（TPAMI 2026） |
| 代码 | <https://github.com/fengyiwu98/RPCANet>（作者官方实现，与 RPCANet 同仓库） |
| 目录 | `RPCANet-main/`（`--net-name rpcanet_pp`） |
| 输入 | Seal 热红外 `T`（单通道） |
| 监督 | 全像素 GT（Seal train 1,500 张） |

RPCANet++ 是 RPCANet 的后续工作（2026，TPAMI），把 RPCA 展开网络扩展为带记忆/长程建模的版本，属于 2026 年的全监督方法。

## 训练协议

| 项目 | 值 |
|---|---|
| 网络 | `get_model('rpcanet_pp')` = `RPCANet_LSTM(stage_num=6, slayers=6, mlayers=3, channel=32)`，官方默认 |
| 损失 | `SoftIoU(out_T)` + 0.01 × `MSE(out_D, input)`；SoftIoU 按 dataset 级（整批累计）计算 |
| 优化器 | Adam，lr **1e-4**（官方 5e-4），poly 衰减 |
| epoch / batch | 60 / 8（官方 50 / 8） |
| 裁剪 | 256×256，随机水平翻转 |
| 模型选择 | Seal val 滑窗像素 IoU@0.5 最优；test 不参与 |
| 测试推理 | 原生分辨率 256 滑窗、步长 128，取稀疏分支 |

## 运行命令

```bash
cd /share_datasets/Seal_Model_Reproduction/RPCANet-main
PY=/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python
$PY train_seal.py --device cuda:1 --net-name rpcanet_pp --epochs 60 --val-interval 5 --out experiments/Seal_fullsup_rpcanetpp
```

## 结果

选中的是 Seal val 上像素 IoU 最高的第 42 epoch 检查点（val IoU 0.33723；训练 60 epoch）。

| 来源 | S↑ | Fw↑ | E↑ | MAE↓ |
|---|---:|---:|---:|---:|
| RPCANet++，val 最优检查点，test 1,983 张 | 0.49800 | 0.51436 | 0.67932 | 0.00028853 |

| IoU↑ | F1↑ | Pd↑ | Fa↓（原始比例） |
|---:|---:|---:|---:|
| 0.30094974 | 0.46266160 | 0.76919951 | 8.81637044101384e-05 |

四项指标由根目录 `evaluate.py` 的四个原函数计算，小目标四项由 `evaluate_small_targets.py` 在同一批预测上计算。完整精度见 `docs/RPCANetpp_Seal评估产物/eight_metrics.json`。

## 复杂度

| Params↓ (M) | GFLOPs↓ | FPS↑ | 单次前向 (ms) |
|---:|---:|---:|---:|
| 2.962 | 190.669 | 80.42 | 12.434 |

单窗 256×256、FP32、batch 1、RTX 5090 实测（thop MACs；三轮 × 每轮 100 次计时，取最好一轮的中位数）。Seal test 一张 622×417 需 12 个滑窗，单图成本约 ×12、单图吞吐约 FPS÷12。对照：RPCANet 仓库 README 的 S6 为 2.915 M / 单图 GPU 0.0471 s，LPCNet 表 2.96 M / 190 G。2.915 vs 2.962 的 1.6 % 差异是 thop 不计 `nn.LSTM` 参数（thop 口径 2.915336 M、全部参数 2.961582 M），模型相同。

## 产物与已知问题

- 权重：`RPCANet-main/experiments/Seal_fullsup_rpcanetpp/{best.pth,last.pth}`；日志 `logs/rpcanetpp_train.log`。
- 预测：`results/RPCANetpp/Seal/fullsup_test_maps/`。
- 与 RPCANet 共用数据与推理脚本，只有 `--net-name` 不同；两行指标来自各自独立训练的权重。
- **训练改动与 RPCANet 相同（仅两条）**：dataset 级 SoftIoU 替代逐样本 SoftIoU、lr 5e-4→1e-4。官方逐样本 soft-IoU + 5e-4 在 Seal 上前 4 个 epoch 冲到 0.30 后发散到 0，本设置稳定在 0.32 左右。网络、epoch、数据与增强不变。
