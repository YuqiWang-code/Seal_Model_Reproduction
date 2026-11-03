# Seal 全监督 IRSTD 基线评估汇总

四个全监督红外小目标检测方法在 Seal 上从零训练（Seal train 1,500 张全像素 GT）、在 Seal test（1,983 张）测试的统一结果。三个方法正常收敛并进入 `docs/Seal量化SOTA表.xlsx` 第 21～23 行；ACM 因官方损失不收敛，只作负结果存档。

## 协议要点

| 项目 | 说明 |
|---|---|
| 输入 | Seal 热红外 `T`：16 位 TIFF 逐图 min-max → 8 位 → 3 通道（`Seal_IRSTD/`） |
| 训练 | Seal train，crop 256×256，batch 8，官方优化器/损失/学习率日程（DNA-Net 250 epoch，MSHNet 400，UIU-Net 500） |
| 选择 | Seal val 500 张滑窗像素 IoU@0.5 最优；test 不参与任何选择 |
| 推理 | 原生分辨率 256×256 滑窗、步长 128，重叠 logits 平均后 sigmoid → 8 位 PNG |
| 四项指标 | 根目录 `evaluate.py` 四个原函数（S / Fw / E / MAE），同一批 1,983 张预测 |
| 小目标四项 | 根目录 `metrics.py` 的 `AS1000Metrics` + `evaluate_small_targets.py`（IoU / F1 / Pd / Fa，原始比例） |

## 八项指标

| 方法 | 会议 | S↑ | Fw↑ | E↑ | MAE↓ | IoU↑ | F1↑ | Pd↑ | Fa↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DNA-Net | TIP22 | 0.49803 | 0.50562 | 0.67979 | 0.00028725 | 0.30000 | 0.46154 | 0.79046 | 8.73e-05 |
| MSHNet | CVPR24 | 0.49818 | 0.54104 | 0.68601 | 0.00024695 | 0.29226 | 0.45232 | 0.80943 | 6.73e-05 |
| UIU-Net | TIP23 | 0.49876 | 0.27240 | 0.68115 | 0.00037482 | 0.21485 | 0.35371 | 0.68739 | 2.96e-05 |
| ACM（未收敛，不入表） | WACV21 | — | — | — | — | — | — | — | — |

## 与弱监督主表的关系

第 3～20 行是跨域泛化（源域训练或公开权重 → Seal test），本节是**同域全监督**（Seal train → Seal test）。两者监督形式、训练数据与难度都不同，只能并列说明"全监督上界/参考"，不能与跨域行直接比大小。

## 产物

| 方法 | 权重 | 预测 | 指标 |
|---|---|---|---|
| DNA-Net | `DNANet-main/experiments/Seal_fullsup/best.pth` | `results/DNANet/Seal/fullsup_test_maps/` | `eight_metrics.json`、`fullsup_metrics.json`、`fullsup_small_metrics.json` |
| MSHNet | `MSHNet-main/experiments/Seal_fullsup/best.pth` | `results/MSHNet/Seal/fullsup_test_maps/` | 同上 |
| UIU-Net | `UIU-Net-main/experiments/Seal_fullsup/best.pth` | `results/UIU-Net/Seal/fullsup_test_maps/` | 同上 |

逐方法的配置、命令与已知问题见 [ACM](ACM_Seal实验记录.md)、[DNA-Net](DNANet_Seal实验记录.md)、[MSHNet](MSHNet_Seal实验记录.md)、[UIU-Net](UIU-Net_Seal实验记录.md)。
