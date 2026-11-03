# DURD：Seal 全监督训练与测试记录

## 方法与来源

| 项目 | 值 |
|---|---|
| 方法 | DURD（Deep Unfolding Residual Decomposition for Infrared Small Target Detection） |
| 论文 | IEEE TGRS 2026 |
| 代码 | <https://github.com/haofan72/DURD>（作者官方实现） |
| 目录 | `DURD-main/` |
| 输入 | Seal 热红外 `T`（单通道） |
| 监督 | 全像素 GT（Seal train 1,500 张） |

来源：awesome-infrared-small-targets 的 Deep Unfolding 条目（2026）。DURD 把低秩-稀疏残差分解展开成网络，`forward` 返回 `L_out, S_out, X_recon, S_prior, M, F_dec, final_pred`，训练用最终预测 `final_pred` 与重构项。

## 训练协议

| 项目 | 值 |
|---|---|
| 网络 | `DURD(in_channel=1)`，官方默认 |
| 损失 | 官方 `DURD_Loss` 等价实现：`SoftIoU(final_pred)` + 0.1 × `MSE(L_out, input)` |
| 优化器 | Adam，lr 1e-3，CosineAnnealingLR 到 1e-5（官方 train.py） |
| epoch / batch | 100 / 8（官方 300 / 8） |
| 裁剪 | 256×256，随机裁剪 + 翻转 |
| 模型选择 | Seal val 滑窗像素 IoU@0.5；test 不参与 |
| 测试推理 | 原生分辨率 256 滑窗，取 `final_pred` |

## 运行命令

```bash
cd /share_datasets/Seal_Model_Reproduction/DURD-main
PY=/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python
$PY train_seal.py --device cuda:1 --epochs 100 --val-interval 10 --batch-size 8 --out experiments/Seal_fullsup
```

## 结果

<!--RESULTS-->

## 产物与已知问题

- 权重：`DURD-main/experiments/Seal_fullsup/{best.pth,last.pth}`；日志 `logs/durd_maintrain.log`。
- 预测：`results/DURD/Seal/fullsup_test_maps/`。
- 官方 `utils/tools.py` 里的 `SoftIoULoss` 是函数而非 `nn.Module`，本实验按其函数签名调用；其余按官方 `train.py`。
- 与官方配置的差异：epoch 300→100、验证改用 Seal val。
