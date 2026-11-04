# HDNet：Seal 全监督训练与测试记录

## 方法与来源

| 项目 | 值 |
|---|---|
| 方法 | HDNet（Hybrid Domain Network with multiscale high-frequency enhancement） |
| 论文 | HDNet: A Hybrid Domain Network With Multiscale High-Frequency Information Enhancement for Infrared Small-Target Detection，IEEE TGRS 2025 |
| 代码 | <https://github.com/xumingzhu989/HDNet-TGRS>（作者官方实现） |
| 目录 | `HDNet-main/` |
| 输入 | Seal 热红外 `T`（3 通道） |
| 监督 | 全像素 GT（Seal train 1,500 张） |

来源：awesome-infrared-small-targets 的 Single-Frame 条目（2025），同时是 PDDiff 论文对比表中的 HDNet。

## 训练协议

| 项目 | 值 |
|---|---|
| 网络 | `HDNet(3)`，多尺度深监督 + 高频增强 |
| 损失 | 官方 `SLSIoULoss`（含形状项），warm-up 5 epoch 后启用，多尺度输出取平均 |
| 优化器 | Adagrad，lr 0.05，CosineAnnealingLR（官方默认） |
| epoch / batch | 120 / 8（官方 800 / 4；缩短到 120 epoch 并按此日程跑完 cosine） |
| 裁剪 | 256×256，官方随机缩放/翻转/裁剪增强 |
| 模型选择 | Seal val 500 张滑窗像素 IoU@0.5 最优；test 不参与 |
| 测试推理 | 原生分辨率 256 滑窗、步长 128，logits 平均后 sigmoid |

## 运行命令

```bash
cd /share_datasets/Seal_Model_Reproduction/HDNet-main
PY=/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python
$PY train_seal.py --device cuda:1 --epochs 120 --val-interval 20 --out experiments/Seal_fullsup
```

## 结果

选中的是 Seal val 上像素 IoU 最高的第 120 epoch 检查点（val IoU 0.28556；120 epoch 日程跑满，每 10 epoch 评估一次，val 一直在 0.27–0.29 平台）。

| 来源 | S↑ | Fw↑ | E↑ | MAE↓ |
|---|---:|---:|---:|---:|
| HDNet，val 最优检查点，test 1,983 张 | 0.49829 | 0.53953 | 0.66410 | 0.00022919 |

| IoU↑ | F1↑ | Pd↑ | Fa↓（原始比例） |
|---:|---:|---:|---:|
| 0.27259038 | 0.42840239 | 0.82134634 | 5.450494617901954e-05 |

四项指标由根目录 `evaluate.py` 的四个原函数计算，小目标四项由 `evaluate_small_targets.py` 在同一批预测上计算。完整精度见 `docs/HDNet_Seal评估产物/eight_metrics.json`。

## 复杂度

| Params↓ (M) | GFLOPs↓ | FPS↑ | 单次前向 (ms) |
|---:|---:|---:|---:|
| 3.676 | 5.697 | 85.54 | 11.691 |

单窗 256×256、FP32、batch 1、RTX 5090 实测（thop MACs；三轮 × 每轮 100 次计时，取最好一轮的中位数）。Seal test 一张 622×417 需 12 个滑窗，单图成本约 ×12、单图吞吐约 FPS÷12。对照：PDDiff 表 3.68 M，参数一致；其 GFLOPs 11.20 G 是 2×MACs 口径，不作比对基准。

## 产物与已知问题

- 权重：`HDNet-main/experiments/Seal_fullsup/{best.pth,last.pth}`；日志 `logs/hdnet_train.log`。
- 预测：`results/HDNet/Seal/fullsup_test_maps/`。
- 官方 `model/HDNet.py` 顶部硬编码 `os.environ['CUDA_VISIBLE_DEVICES']="0"`，会让进程只看到 0 号卡；本实验删掉该行并注释说明，改为用 `--device` 选卡。其余网络结构未改。
- 与官方配置的差异：epoch 800→120（跑满 120 epoch）、batch 4→8、验证改用 Seal val（官方脚本以 test 选模型）。
