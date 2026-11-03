# DRPCA-Net：Seal 全监督训练与测试记录

## 方法与来源

| 项目 | 值 |
|---|---|
| 方法 | DRPCA-Net（Make Robust PCA Great Again for Infrared Small Target Detection） |
| 论文 | IEEE TGRS 2025，GrokCV 组工作 |
| 代码 | <https://github.com/GrokCV/DRPCA-Net>（作者官方实现） |
| 目录 | `DRPCA-Net-main/`（`Drpcanet`，stage_num=6） |
| 输入 | Seal 热红外 `T`（单通道） |
| 监督 | 全像素 GT（Seal train 1,500 张） |

来源：GrokCV 出版物页（TGRS 2025）+ awesome-infrared-small-targets 的 Deep Unfolding 条目。与 RPCANet 同属 RPCA 展开网络，输出低秩背景 `out_D` 与稀疏目标 `out_T`。

## 训练协议

| 项目 | 值 |
|---|---|
| 网络 | `get_model('Drpcanet')` = `DRPCANet(stage_num=6)`，官方默认 |
| 损失 | dataset 级 SoftIoU（`out_T`）+ 0.01 × `MSE(out_D, input)`（与 RPCANet 同构，见说明） |
| 优化器 | Adam，lr 1e-4，poly 衰减 |
| epoch / batch | 60 / 8 |
| 裁剪 | 256×256，随机翻转 |
| 模型选择 | Seal val 滑窗像素 IoU@0.5；test 不参与 |
| 测试推理 | 原生分辨率 256 滑窗，取稀疏分支 |

## 运行命令

```bash
cd /share_datasets/Seal_Model_Reproduction/DRPCA-Net-main
PY=/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python
$PY train_seal.py --device cuda:1 --epochs 60 --val-interval 5 --batch-size 8 --out experiments/Seal_fullsup
```

## 结果

<!--RESULTS-->

## 产物与已知问题

- 权重：`DRPCA-Net-main/experiments/Seal_fullsup/{best.pth,last.pth}`；日志 `logs/drpca_net_maintrain.log`。
- 预测：`results/DRPCANet/Seal/fullsup_test_maps/`。
- 官方 `run_config.py` 把 lr 设为 1e-5、epoch 400；本实验按 RPCANet 系列在 Seal 上已验证的设置（dataset 级 SoftIoU + Adam 1e-4 + 60 epoch）训练，网络结构与损失组成不变。
