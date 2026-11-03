# UIU-Net：Seal 全监督训练与测试记录

## 方法与来源

| 项目 | 值 |
|---|---|
| 方法 | UIU-Net（U-Net in U-Net） |
| 论文 | UIU-Net: U-Net in U-Net for Infrared Small Object Detection，IEEE TIP 2023 |
| 代码 | <https://github.com/danfenghong/IEEE_TIP_UIU-Net>（作者官方实现） |
| 目录 | `UIU-Net-main/` |
| 输入模态 | Seal 热红外 `T`（单通道复制为 3 通道） |
| 监督 | 全像素 GT（Seal train 1,500 张） |

UIU-Net 用两层嵌套 U-Net 与多尺度融合提升小目标响应，是 IRSTD 常用全监督基线之一（awesome-infrared-small-targets 收录）。其官方仓库的发布训练脚本面向 DUTS 显著目标数据，本实验按论文的 SIRST 设置（3 通道输入、7 个深监督输出、BCE）在 Seal 上重写训练入口。

## 数据准备

与 ACM 完全一致：`Seal_IRSTD/`，热红外 16 位逐图 min-max 拉伸到 8 位并复制 3 通道，GT 二值化，划分 1,500 / 500 / 1,983。转换细节与 SNR 结论见 [ACM_Seal实验记录.md](ACM_Seal实验记录.md#数据准备)。

## 训练协议

| 项目 | 值 |
|---|---|
| 网络 | `UIUNET(3, 1)`，官方结构，输出 7 个 sigmoid 概率图 |
| 损失 | 7 个输出的 BCE 之和（官方 `muti_bce_loss_fusion` 等价实现） |
| 优化器 | Adam，lr 1e-3，betas=(0.9, 0.999)，eps=1e-8（官方默认） |
| epoch / batch | 500 / 8（官方 500 / 3；利用 32 GB 显存把 batch 提到 8） |
| 裁剪 | 256×256（官方 `RescaleT(320) + RandomCrop(288)`；Seal 原生 417×622，改为在原生尺度上直接裁剪 256，避免缩放改变目标像素数） |
| 增强 | 随机水平翻转（官方同样只有翻转，无颜色抖动） |
| 归一化 | 官方 `ToTensorLab(flag=0)`：逐样本除以最大值后按 ImageNet 均值/方差标准化（三通道同值） |
| 模型选择 | Seal **val** 500 张，滑窗推理像素 IoU@0.5，取最佳 `best.pth`；test 不参与选择 |
| 测试推理 | 原生分辨率 256×256 滑窗、步长 128；把各窗口的 `logit(prob)` 平均后再 sigmoid，输出 8 位 PNG |

训练入口 `UIU-Net-main/train_seal.py`，推理入口 `UIU-Net-main/infer_seal.py`，数据为 `seal_dataset.py`，滑窗为 `seal_infer.py`。

## 运行命令

```bash
cd /share_datasets/Seal_Model_Reproduction/UIU-Net-main
PY=/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python

$PY train_seal.py --device cuda:0 --epochs 500 --val-interval 20 --out experiments/Seal_fullsup

cd /share_datasets/Seal_Model_Reproduction
$PY evaluate.py --pred_root results/UIU-Net/Seal/fullsup_test_maps \
  --gt_dir /share_datasets/Seal_Model_Reproduction/Seal_data/test/GT \
  --json_output results/UIU-Net/Seal/fullsup_metrics.json
$PY evaluate_small_targets.py --pred_root results/UIU-Net/Seal/fullsup_test_maps \
  --gt_dir /share_datasets/Seal_Model_Reproduction/Seal_data/test/GT \
  --json_output results/UIU-Net/Seal/fullsup_small_metrics.json
```

## 结果

选中的是 Seal val 上像素 IoU 最高的第 20 epoch 检查点（val IoU 0.23816）。此后 BCE 迅速退化为"全背景"解：第 40/60/80/100 epoch 的 val IoU 为 0.000/0.002/0.000/0.000，训练损失降到 0.005 并停滞。按 val 连续 5 次评估无改善的规则，在第 114 epoch 停止训练（只保存了 val 最优的 `best.pth`）。

| 来源 | S↑ | Fw↑ | E↑ | MAE↓ |
|---|---:|---:|---:|---:|
| UIU-Net，val 最优检查点（第 20 epoch），test 1,983 张 | 0.49876 | 0.27240 | 0.68115 | 0.00037482 |

| IoU↑ | F1↑ | Pd↑ | Fa↓（原始比例） |
|---:|---:|---:|---:|
| 0.21485370 | 0.35371124 | 0.68738995 | 2.9647782131835235e-05 |

四项指标由根目录 `evaluate.py` 的四个原函数计算，小目标四项由 `evaluate_small_targets.py` 在同一批预测上计算。完整精度见 `docs/UIU-Net_Seal评估产物/eight_metrics.json`。

**关于退化的说明**：UIU-Net 官方训练只监督 7 个输出的 BCE，Seal 的前景占 0.02%，BCE 的全局最优就是"全背景"，因此第 20 epoch 之后模型滑向该不动点。这与 ACM 的现象一致（加入 BCE 的对照实验也验证了 BCE 偏好全背景）。表中保留 val 最优检查点的真实数值，不为了好看而换损失重训。

## 产物与已知问题

- 权重：`UIU-Net-main/experiments/Seal_fullsup/{best.pth,last.pth}`；日志 `logs/uiunet_train.log`，逐 epoch 记录 `validation.jsonl`。
- 预测：`results/UIU-Net/Seal/fullsup_test_maps/`；指标 JSON 同目录。
- 与官方配置的差异：batch 3→8、裁剪改为原生 256（官方先缩放再裁 288）、验证集为 Seal val、输入为 Seal 热红外单模态。epoch、结构、损失、优化器与官方一致。
- 官方 `train.py` 的 DUTS 数据路径、`save_frq` 等训练代码未修改，仅新增 `train_seal.py`/`infer_seal.py`，历史脚本保持原样。
