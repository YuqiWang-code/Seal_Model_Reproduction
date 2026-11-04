# BGM：Seal 全监督训练与测试记录

## 方法与来源

| 项目 | 值 |
|---|---|
| 方法 | BGM（Forgetting the Background: A Masking Approach for Enhanced Infrared Small-Target Detection） |
| 论文 | TGRS 2025 |
| 代码 | <https://github.com/ZhihaoMa123/BGM>（作者官方实现） |
| 目录 | `BGM-main/`（`BasicUNet_plus` 主干） |
| 输入 | Seal 热红外 `T`（单通道） |
| 监督 | 全像素 GT（Seal train 1,500 张） |

来源：awesome-infrared-small-targets 的 Single-Frame 条目（2025）。方法通过背景掩码让网络"忘记背景"。

## 训练协议

| 项目 | 值 |
|---|---|
| 网络 | `Net(model_name='BasicUNet_plus')`，官方默认组合 |
| 损失 | 官方 `net.loss(pred, gt)`（`loss.py` 中的组合损失） |
| 优化器 | Adam，lr 5e-4，MultiStepLR [100,140] ×0.1（官方 train.py 设置） |
| epoch / batch | 150 / 8（官方 400 / 16） |
| 裁剪 | 256×256，官方随机裁剪 + 翻转 |
| 模型选择 | Seal val 滑窗像素 IoU@0.5 最优；test 不参与 |
| 测试推理 | 原生分辨率 256 滑窗、步长 128 |

## 运行命令

```bash
cd /share_datasets/Seal_Model_Reproduction/BGM-main
PY=/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python
$PY train_seal.py --device cuda:0 --epochs 150 --val-interval 10 --batch-size 8 --out experiments/Seal_fullsup
```

## 结果

选中的是 Seal val 上像素 IoU 最高的第 50 epoch 检查点（val IoU 0.34709；训练跑满 150 epoch）。

| 来源 | S↑ | Fw↑ | E↑ | MAE↓ |
|---|---:|---:|---:|---:|
| BGM，val 最优检查点，test 1,983 张 | 0.49811 | 0.49902 | 0.66520 | 0.00030535 |

| IoU↑ | F1↑ | Pd↑ | Fa↓（原始比例） |
|---:|---:|---:|---:|
| 0.30734902 | 0.47018663 | 0.78680753 | 9.631786522467818e-05 |

四项指标由根目录 `evaluate.py` 的四个原函数计算，小目标四项由 `evaluate_small_targets.py` 在同一批预测上计算。完整精度见 `docs/BGM_Seal评估产物/eight_metrics.json`。

## 产物与已知问题

- 权重：`BGM-main/experiments/Seal_fullsup/{best.pth,last.pth}`；日志 `logs/bgm_maintrain.log`。
- 预测：`results/BGM/Seal/fullsup_test_maps/`。
- 与官方配置的差异：epoch 400→150、batch 16→8、验证改用 Seal val（官方脚本用 test 选模型）。
