# DHiF：Seal 全监督训练与测试记录

## 方法与来源

| 项目 | 值 |
|---|---|
| 方法 | DHiF（Dynamic High-frequency Convolution），MTU 主干 |
| 论文 | Dynamic High-frequency Convolution for Infrared Small Target Detection，IEEE TCSVT 2026 |
| 代码 | <https://github.com/TinaLRJ/DHiF>（作者官方实现） |
| 目录 | `DHiF-main/`（`--model MTU --conv DHiF`） |
| 输入 | Seal 热红外 `T`（单通道） |
| 监督 | 全像素 GT（Seal train 1,500 张） |

来源：awesome-infrared-small-targets 的 Single-Frame 条目（2026）。DHiF 是把高频动态卷积作为插件嵌入 IRSTD 主干，官方默认用 MTU-Net 主干。

## 训练协议

| 项目 | 值 |
|---|---|
| 网络 | `Net(model_name='MTU', mode='train', conv='DHiF')`，官方默认组合 |
| 损失 | 官方 `SoftIoULoss`（`net.loss`） |
| 优化器 | Adam，lr 5e-4，MultiStepLR 每 50 epoch ×0.5（官方默认） |
| epoch / batch | 100 / 12（官方 500 / 16；按本项目预算缩减） |
| 裁剪 | 256×256，官方随机裁剪 + 翻转增强 |
| 模型选择 | Seal val 500 张滑窗像素 IoU@0.5 最优；test 不参与 |
| 测试推理 | 原生分辨率 256 滑窗、步长 128 |

## 运行命令

```bash
cd /share_datasets/Seal_Model_Reproduction/DHiF-main
PY=/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python
$PY train_seal.py --device cuda:1 --epochs 100 --val-interval 10 --batch-size 12 --out experiments/Seal_fullsup
```

## 结果

<!--RESULTS-->

## 产物与已知问题

- 权重：`DHiF-main/experiments/Seal_fullsup/{best.pth,last.pth}`；日志 `logs/dhif_train.log`。
- 预测：`results/DHiF/Seal/fullsup_test_maps/`。
- 官方压缩包内部不一致：`model/Unet/utils.py` 引用了 `model/basic.py` 里已不存在的 `DualConv_FC_light`，导致 `import model` 直接报错。本实验只保留 MTU 主干，把 `model/__init__.py` 的其他主干导入删掉（原文件已注释说明），其余代码未动。
- 依赖：服务器环境补装 `PyWavelets`（`model/wtconv` 需要 `pywt`）。
- 与官方配置的差异：epoch 500→100、batch 16→12、输入为 Seal 热红外、验证改用 Seal val（官方把 test 当验证集并按 test IoU 存权重）。
