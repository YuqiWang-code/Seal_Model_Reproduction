# ACM：Seal 全监督训练记录（未进入主表）

## 结论

ACM 使用官方 SoftIoU 损失在 Seal 上无法收敛：训练损失停在 0.66 左右，滑窗推理输出恒为背景（186 个 epoch 的验证像素 IoU 始终为 0），因此**不进入量化对比表**。本目录保留代码、训练入口和两次随机种子的实验记录，作为负结果存档。同期训练的 DNA-Net、MSHNet 正常收敛，UIU-Net 在第 20 epoch 后滑向全背景解，三者（含 UIU-Net 的 val 最优检查点）数值见各自实验记录。

## 方法与来源

| 项目 | 值 |
|---|---|
| 方法 | ACM（Asymmetric Contextual Modulation） |
| 论文 | Asymmetric Contextual Modulation for Infrared Small Target Detection，WACV 2021 |
| 代码 | <https://github.com/Tianfang-Zhang/acm-pytorch>（社区 PyTorch 实现，对应官方 MXNet 版 open-acm） |
| 目录 | `ACM-main/` |
| 输入 | Seal 热红外 `T`（单模态，复制为 3 通道） |
| 监督 | 全像素 GT（Seal train 1,500 张） |

## 训练配置

| 项目 | 值 |
|---|---|
| 网络 | `ASKCResNetFPN([4,4,4], [8,16,32,64], 'AsymBi')` |
| 损失 | 官方 `SoftLoULoss`（逐样本 soft-IoU） |
| 优化器 | Adagrad，lr 0.05，weight decay 1e-4，官方 poly 衰减 |
| epoch / batch | 300 / 8 |
| 裁剪 | 256×256（官方 480） |
| 增强 | 官方：随机翻转、长边 0.5–2.0× 缩放、随机裁剪、50% 高斯模糊 |
| 稳定化尝试 | ①官方 5 epoch warm-up（`--warm-up-epochs 5`，走官方调度器自带的分支）；②两个随机种子（seed 1、2） |
| 模型选择 | Seal val 500 张，滑窗像素 IoU@0.5 |

## 观测证据

| 现象 | 数值 |
|---|---|
| 训练损失 | 0.9996（初始）→ 0.66 左右并停滞 |
| val 像素 IoU（ep 20/40/…/180） | 全部 0.00000（seed 1、2 相同） |
| 推理输出 | 全图为背景，logit 均值 −23 至 −26，最大 logit 仍为负 |
| 单图过拟合测试 | 失败：单张 256 裁剪训练 200 步后，在 0 附近振荡、IoU 反复掉回 0 |

对照实验：把 16 位热红外按 1–99 百分位裁剪拉伸（背景噪声放大 10 倍、目标 SNR 1.5）时，ACM 同样退化；改为逐图 min-max（SNR 2.2–29.7）后 DNANet/MSHNet/UIU-Net 均正常，ACM 仍不收敛，说明问题在 ACM 的损失/容量组合，而非数据管线。

## 为什么放弃

- ACM 的逐样本 soft-IoU 在 Seal 上属于极端稀疏（前景 0.02%），梯度被大量无目标样本稀释；Adagrad lr 0.05 的更新又足够大，训练容易落到"全背景"不动点。
- 加入 BCE 项会更糟：BCE 的最优解同样是全背景（单图过拟合测试里 BCE 降到 0.0036 但 IoU 为 0）。
- 该退化与 test 集无关（只用 train/val 观测），也没有为凑数值而调整超参，因此按协议不再纳入主表，避免用一个明显未收敛的模型充当"对比方法"。

## 运行命令（存档）

```bash
cd /share_datasets/Seal_Model_Reproduction/ACM-main
PY=/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python
$PY train_seal.py --device cuda:1 --epochs 300 --val-interval 20 --seed 1 \
  --out experiments/Seal_fullsup_seed1
$PY train_seal.py --device cuda:1 --epochs 300 --val-interval 20 --seed 2 \
  --out experiments/Seal_fullsup_seed2
```

日志：`logs/acm_seed1.log`、`logs/acm_seed2.log`（服务器）。产物：`ACM-main/experiments/Seal_fullsup_seed{1,2}/`。

## 复用说明

`seal_dataset.py`、`seal_infer.py`、`train_seal.py`、`infer_seal.py` 与其余三个基线同构，可直接用于后续换损失或换初始化重试；本次不产出 test 预测，也不往表里填任何 ACM 数值。
