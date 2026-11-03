# SCTransNet：Seal 全监督训练与测试记录

## 方法与来源

| 项目 | 值 |
|---|---|
| 方法 | SCTransNet（Spatial-channel Cross Transformer Network） |
| 论文 | SCTransNet: Spatial-channel Cross Transformer Network for Infrared Small Target Detection，IEEE TGRS 2024 |
| 代码 | <https://github.com/xdFai/SCTransNet>（作者官方实现） |
| 目录 | `SCTransNet-main/` |
| 输入 | Seal 热红外 `T`（单通道） |
| 监督 | 全像素 GT（Seal train 1,500 张） |

来源：awesome-infrared-small-targets 的 Single-Frame 全监督条目，也出现在 PDDiff 论文（arXiv 版）的对比表中。

## 数据准备

`Seal_IRSTD/`：热红外 16 位逐图 min-max → 8 位三通道 PNG（本方法取单通道），GT 二值化，划分 1,500 / 500 / 1,983。SCTransNet 官方用 `(img - mean) / std` 归一化，本实验把 mean/std 换成 Seal train 前 300 张图的实测值并缓存在 `norm_stats.json`。

## 训练协议

| 项目 | 值 |
|---|---|
| 网络 | `SCTransNet(config, n_channels=1, n_classes=1, img_size=256, deepsuper=True)`，config 为官方 `get_SCTrans_config()`（base_channel 32、patch [16,8,4,2]） |
| 损失 | 6 个深监督输出的 BCE 均值（官方 `Net.loss` 等价实现） |
| 优化器 | Adam，lr 1e-3，CosineAnnealingLR 到 1e-5（官方默认） |
| epoch / batch | 300 / 8（官方 1,000 / 16；按本项目预算缩减） |
| 裁剪 | 256×256 |
| 增强 | 随机水平翻转 + 随机裁剪（官方 `random_crop` + `augumentation`） |
| 模型选择 | Seal val 500 张，滑窗像素 IoU@0.5 最优；test 不参与 |
| 测试推理 | 原生分辨率 256 滑窗、步长 128，重叠区 logits 平均后 sigmoid |

## 运行命令

```bash
cd /share_datasets/Seal_Model_Reproduction/SCTransNet-main
PY=/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python
$PY train_seal.py --device cuda:0 --epochs 300 --val-interval 20 --batch-size 8 --out experiments/Seal_fullsup
```

## 结果

选中的是 Seal val 上像素 IoU 最高的第 280 epoch 检查点（val IoU 0.26800；训练跑满 300 epoch，val 在 0.20–0.27 间波动）。测试集 1,983 张预测全部生成。

| 来源 | S↑ | Fw↑ | E↑ | MAE↓ |
|---|---:|---:|---:|---:|
| SCTransNet，val 最优检查点，test 1,983 张 | 0.49902 | 0.32255 | 0.67917 | 0.00030830 |

| IoU↑ | F1↑ | Pd↑ | Fa↓（原始比例） |
|---:|---:|---:|---:|
| 0.20290176 | 0.33735383 | 0.62210484 | 2.5279065071684813e-05 |

四项指标由根目录 `evaluate.py` 的四个原函数计算，小目标四项由 `evaluate_small_targets.py` 在同一批预测上计算。完整精度见 `docs/SCTransNet_Seal评估产物/eight_metrics.json`。

## 产物与已知问题

- 权重：`SCTransNet-main/experiments/Seal_fullsup/{best.pth,last.pth}`；日志 `logs/sctransnet_train.log`。
- 预测：`results/SCTransNet/Seal/fullsup_test_maps/`；指标 JSON 同目录。
- 依赖：服务器环境补装 `ml_collections`、`einops`、`thop`（官方 `model/Config.py`、`model/SCTransNet.py` 需要）。
- 与官方配置的差异：epoch 1,000→300、batch 16→8、输入改为 Seal 热红外单通道、验证改用 Seal val（官方把 test 当验证集）。
