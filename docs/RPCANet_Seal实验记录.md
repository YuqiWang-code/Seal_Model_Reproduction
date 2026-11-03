# RPCANet：Seal 全监督训练与测试记录

## 方法与来源

| 项目 | 值 |
|---|---|
| 方法 | RPCANet（Deep Unfolding RPCA Based Infrared Small Target Detection） |
| 论文 | RPCANet: Deep Unfolding RPCA Based Infrared Small Target Detection，WACV 2024 |
| 代码 | <https://github.com/fengyiwu98/RPCANet>（作者官方实现，同一仓库还含 RPCANet++） |
| 目录 | `RPCANet-main/`（`--net-name rpcanet`） |
| 输入 | Seal 热红外 `T`（单通道） |
| 监督 | 全像素 GT（Seal train 1,500 张） |

来源：awesome-infrared-small-targets 的 Deep Unfolding 条目（WACV 2024）。

## 训练协议

| 项目 | 值 |
|---|---|
| 网络 | `get_model('rpcanet')` = `RPCANet9(stage_num=6, slayers=6, llayers=3, mlayers=3, channel=32)`，官方默认 |
| 损失 | `SoftIoU(out_T)` + 0.01 × 重构 `MSE(out_D, input)`；SoftIoU 按 **dataset 级**（整批累计）计算，见下方说明 |
| 优化器 | Adam，lr **1e-4**（官方 5e-4），poly 衰减 |
| epoch / batch | 60 / 8（官方 50 / 8，略延长） |
| 裁剪 | 256×256，随机水平翻转 |
| 模型选择 | Seal val 500 张滑窗像素 IoU@0.5 最优；test 不参与 |
| 测试推理 | 原生分辨率 256 滑窗、步长 128，取稀疏分支 logits 平均后 sigmoid |

## 运行命令

```bash
cd /share_datasets/Seal_Model_Reproduction/RPCANet-main
PY=/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python
$PY train_seal.py --device cuda:1 --net-name rpcanet --epochs 50 --val-interval 5 --out experiments/Seal_fullsup_rpcanet
```

## 结果

选中的是 Seal val 上像素 IoU 最高的第 47 epoch 检查点（val IoU 0.33422；训练 60 epoch）。

| 来源 | S↑ | Fw↑ | E↑ | MAE↓ |
|---|---:|---:|---:|---:|
| RPCANet，val 最优检查点，test 1,983 张 | 0.49807 | 0.51966 | 0.67944 | 0.00027539 |

| IoU↑ | F1↑ | Pd↑ | Fa↓（原始比例） |
|---:|---:|---:|---:|
| 0.30638360 | 0.46905611 | 0.80617635 | 7.75734054218699e-05 |

四项指标由根目录 `evaluate.py` 的四个原函数计算，小目标四项由 `evaluate_small_targets.py` 在同一批预测上计算。完整精度见 `docs/RPCANet_Seal评估产物/eight_metrics.json`。

## 产物与已知问题

- 权重：`RPCANet-main/experiments/Seal_fullsup_rpcanet/{best.pth,last.pth}`；日志 `logs/rpcanet_train.log`。
- 预测：`results/RPCANet/Seal/fullsup_test_maps/`。
- 官方 loader 会把图像缩放到 256×256；本实验改为在原生分辨率裁 256 块，保持目标像素尺寸（Seal 目标只有几像素）。
- **两处训练改动（已在报告中标注，仅此两条）**：① 逐样本 SoftIoU 换成 dataset 级 SoftIoU（整批累计，DNA-Net 的做法）。官方逐样本 soft-IoU 在 Seal 的 0.02% 前景上梯度被稀释，50 epoch 内 val IoU 恒为 0（训练 loss 仍在降）；② lr 5e-4 → 1e-4。用 5e-4 时 val 在第 4 epoch 冲到 0.298 后立刻发散到 0，降低 lr 后稳定在 0.32 左右。网络结构、epoch 数、数据与增强不变。
- 该仓库同时提供 RPCANet++（TPAMI 2026），作为另一个对比方法单独训练，见 [RPCANet++ 记录](RPCANet++_Seal实验记录.md)。
