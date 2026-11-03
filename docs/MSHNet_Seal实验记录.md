# MSHNet：Seal 全监督训练与测试记录

## 方法与来源

| 项目 | 值 |
|---|---|
| 方法 | MSHNet（Multi-Scale and Location Sensitive Network） |
| 论文 | Infrared Small Target Detection with Scale and Location Sensitivity，CVPR 2024 |
| 代码 | <https://github.com/Lliu666/MSHNet>（作者官方实现） |
| 目录 | `MSHNet-main/` |
| 输入模态 | Seal 热红外 `T`（单通道复制为 3 通道） |
| 监督 | 全像素 GT（Seal train 1,500 张） |

MSHNet 是较新的全监督 IRSTD 方法（CVPR 2024），采用尺度感知与位置敏感损失 SLSIoU 及多尺度深监督，在 awesome-infrared-small-targets 的 “Single-Frame” 中收录。

## 数据准备

与 ACM 完全一致：`Seal_IRSTD/`，热红外 16 位逐图 min-max 拉伸到 8 位并复制 3 通道，GT 二值化，划分 1,500 / 500 / 1,983。转换细节与 SNR 结论见 [ACM_Seal实验记录.md](ACM_Seal实验记录.md#数据准备)。

## 训练协议

| 项目 | 值 |
|---|---|
| 网络 | `MSHNet(3)`，官方默认结构 |
| 损失 | 官方 `SLSIoULoss`（含 shape/长度-角度项），warm-up 5 epoch 后启用，多尺度输出按官方权重平均 |
| 优化器 | Adagrad，lr 0.05（官方默认，无 weight decay） |
| epoch / batch | 400 / 8（官方 400 / 4；利用 32 GB 显存把 batch 提到 8） |
| 裁剪 | 256×256（官方默认） |
| 增强 | 官方 `IRSTD_Dataset`：随机水平翻转、长边 0.5–2.0× 随机缩放、随机裁剪、50% 高斯模糊 |
| 模型选择 | Seal **val** 500 张，滑窗推理像素 IoU@0.5，取最佳 `best.pth`；test 不参与选择（官方脚本用 test 选模型） |
| 测试推理 | 原生分辨率 256×256 滑窗、步长 128，重叠区 logits 平均后 sigmoid，输出 8 位 PNG |

训练入口 `MSHNet-main/train_seal.py`，推理入口 `MSHNet-main/infer_seal.py`。

## 运行命令

```bash
cd /share_datasets/Seal_Model_Reproduction/MSHNet-main
PY=/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python

$PY train_seal.py --device cuda:0 --epochs 400 --val-interval 20 --out experiments/Seal_fullsup

cd /share_datasets/Seal_Model_Reproduction
$PY evaluate.py --pred_root results/MSHNet/Seal/fullsup_test_maps \
  --gt_dir /share_datasets/Seal_Model_Reproduction/Seal_data/test/GT \
  --json_output results/MSHNet/Seal/fullsup_metrics.json
$PY evaluate_small_targets.py --pred_root results/MSHNet/Seal/fullsup_test_maps \
  --gt_dir /share_datasets/Seal_Model_Reproduction/Seal_data/test/GT \
  --json_output results/MSHNet/Seal/fullsup_small_metrics.json
```

## 结果

选中的是 Seal val 上像素 IoU 最高的第 140 epoch 检查点（val IoU 0.30430；训练跑满 400 epoch，末端 val IoU 0.28487，从第 160 epoch 起开始过拟合）。测试集 1,983 张预测全部生成，文件名与 `Seal_data/test/GT` 一一对应。

| 来源 | S↑ | Fw↑ | E↑ | MAE↓ |
|---|---:|---:|---:|---:|
| MSHNet，val 最优检查点，test 1,983 张 | 0.49818 | 0.54104 | 0.68601 | 0.00024695 |

| IoU↑ | F1↑ | Pd↑ | Fa↓（原始比例） |
|---:|---:|---:|---:|
| 0.29225813 | 0.45232160 | 0.80942706 | 6.727668733083446e-05 |

四项指标由根目录 `evaluate.py` 的四个原函数计算，小目标四项由 `evaluate_small_targets.py` 在同一批预测上计算。完整精度见 `docs/MSHNet_Seal评估产物/eight_metrics.json`。

## 产物与已知问题

- 权重：`MSHNet-main/experiments/Seal_fullsup/{best.pth,last.pth}`；日志 `logs/mshnet_train.log`，逐 epoch 记录 `validation.jsonl`。
- 预测：`results/MSHNet/Seal/fullsup_test_maps/`；指标 JSON 同目录。
- 与官方配置的差异：batch 4→8、验证集改为 Seal val、输入为 Seal 热红外单模态。epoch、结构、损失、优化器、学习率与官方一致。
- 服务器环境需补装 `scikit-image`（官方 `model/loss.py` 顶部 `from skimage import measure` 未被使用但会阻断导入），已安装 0.25.2。
