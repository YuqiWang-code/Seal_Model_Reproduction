# DNA-Net：Seal 全监督训练与测试记录

## 方法与来源

| 项目 | 值 |
|---|---|
| 方法 | DNA-Net（Dense Nested Attention Network） |
| 论文 | Dense Nested Attention Network for Infrared Small Target Detection，IEEE TIP 2022 |
| 代码 | <https://github.com/YeRen123455/Infrared-Small-Target-Detection>（作者官方实现） |
| 目录 | `DNANet-main/` |
| 输入模态 | Seal 热红外 `T`（单通道复制为 3 通道） |
| 监督 | 全像素 GT（Seal train 1,500 张） |

DNA-Net 是 IRSTD 领域最常用的全监督基线之一，在 awesome-infrared-small-targets 的 “Single-Frame” 中列为 DNANet，也是 NUDT-SIRST / IRSTD-1k 两个标准数据集的原作者代码。

## 数据准备

与 ACM 完全一致：`Seal_IRSTD/` 由 `Seal_data` 生成，热红外 16 位逐图 min-max 拉伸到 8 位并复制 3 通道，GT 二值化，划分 1,500 / 500 / 1,983。转换细节与 SNR 结论见 [ACM_Seal实验记录.md](ACM_Seal实验记录.md#数据准备)。

## 训练协议

| 项目 | 值 |
|---|---|
| 网络 | `DNANet(num_classes=1, input_channels=3, block=Res_CBAM_block, num_blocks=[2,2,2,2], nb_filter=[16,32,64,128,256], deep_supervision='True')`，即官方 `channel_size=three, backbone=resnet_18` |
| 损失 | 官方 `SoftIoULoss`，对 4 个深监督输出求和取平均 |
| 优化器 | Adagrad，lr 0.05，CosineAnnealingLR（T_max=epochs，eta_min=1e-5），官方默认 |
| epoch / batch | 250 / 8（官方 1,500 / 4；为与其余基线同等时间预算，降到 250 epoch 跑完完整 cosine 日程，并利用显存把 batch 提到 8） |
| 裁剪 | 256×256（官方默认） |
| 增强 | 官方 `TrainSetLoader`：随机水平翻转、长边 0.5–2.0× 随机缩放、随机裁剪、50% 高斯模糊 |
| 模型选择 | Seal **val** 500 张，滑窗推理像素 IoU@0.5，取最佳 `best.pth`；test 不参与选择（官方脚本用 test 选模型，这里改为 val） |
| 测试推理 | 原生分辨率 256×256 滑窗、步长 128，重叠区 logits 平均后 sigmoid，输出 8 位 PNG |

训练入口 `DNANet-main/train_seal.py`，推理入口 `DNANet-main/infer_seal.py`，数据与滑窗实现为 `seal_dataset.py`、`seal_infer.py`。

## 运行命令

```bash
cd /share_datasets/Seal_Model_Reproduction/DNANet-main
PY=/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python

$PY train_seal.py --device cuda:1 --epochs 250 --val-interval 20 --out experiments/Seal_fullsup

cd /share_datasets/Seal_Model_Reproduction
$PY evaluate.py --pred_root results/DNANet/Seal/fullsup_test_maps \
  --gt_dir /share_datasets/Seal_Model_Reproduction/Seal_data/test/GT \
  --json_output results/DNANet/Seal/fullsup_metrics.json
$PY evaluate_small_targets.py --pred_root results/DNANet/Seal/fullsup_test_maps \
  --gt_dir /share_datasets/Seal_Model_Reproduction/Seal_data/test/GT \
  --json_output results/DNANet/Seal/fullsup_small_metrics.json
```

## 结果

选中的是 Seal val 上像素 IoU 最高的第 120 epoch 检查点（val IoU 0.33819；训练在第 250 epoch 结束，末端 val IoU 0.33398）。测试集 1,983 张预测全部生成，文件名与 `Seal_data/test/GT` 一一对应。

| 来源 | S↑ | Fw↑ | E↑ | MAE↓ |
|---|---:|---:|---:|---:|
| DNA-Net，val 最优检查点，test 1,983 张 | 0.49803 | 0.50562 | 0.67979 | 0.00028725 |

| IoU↑ | F1↑ | Pd↑ | Fa↓（原始比例） |
|---:|---:|---:|---:|
| 0.30000332 | 0.46154239 | 0.79046458 | 8.7298515673e-05 |

四项指标由根目录 `evaluate.py` 的四个原函数计算，小目标四项由 `evaluate_small_targets.py` 在同一批预测上计算。完整精度见 `docs/DNANet_Seal评估产物/eight_metrics.json`，原始两份 JSON 同目录。

作为对照：Ours（RGB-T、框监督）S 0.49915、Fw 0.67338、E 0.66731、MAE 0.00015、IoU 0.44835、F1 0.61912、Pd 0.82731、Fa 2.10e-06。DNA-Net 全监督在 test 上的 IoU/Pd 接近 Ours，但 Fw 明显更低、Fa 高约 40 倍，说明其预测的连通域更"散"，小目标级惩罚更重。

## 产物与已知问题

- 权重：`DNANet-main/experiments/Seal_fullsup/{best.pth,last.pth}`；日志 `logs/dnanet_train.log`，逐 epoch 记录 `validation.jsonl`。
- 预测：`results/DNANet/Seal/fullsup_test_maps/`；指标 JSON 同目录。
- 与官方配置的差异：epoch 1,500→250、batch 4→8、验证集改为 Seal val、输入为 Seal 热红外单模态。网络结构、损失、优化器、学习率日程与官方一致。
- 仓库自带的 `pretrain_DNANet_model.tar` 是在 NUDT-SIRST 上训练好的权重，本实验**没有**使用它：按本项目协议，无 Seal 参与的全监督基线必须在 Seal train 上训练，从零初始化（`weights_init_xavier`）。
