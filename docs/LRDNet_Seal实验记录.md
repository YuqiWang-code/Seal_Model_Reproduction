# LRDNet：S-COD 源域重训后的 Seal 跨域测试

主表采用源域 S-COD/CAMO 涂鸦监督训练的 LRDNet，在 Seal test 上直接测试。作者完整方法权重链接失效，复用已完成的源域两阶段训练；权重与预测可用，本次无需重新训练或推理。

## 主表八项指标

| S↑ | Fw↑ | E↑ | MAE↓ | IoU↑ | F1↑ | Pd↑ | Fa↓（原始比例） |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.48582 | 0.00145 | 0.27485 | 0.01099 | 0.00064195 | 0.00128307 | 0.00067723 | 0.0107905600 |

八项均来自 `experiments/SCOD/test_maps/` 的同一批 1,983 张原尺寸灰度预测。主表方法名为 `LRDNet-S-COD`，协议注明“S-COD源域重训后跨域测试”。

原四项使用项目根目录 `evaluate.py` 的四个固定原函数，其函数源码与 `Ours/evaluate.py` 一致。E 对连续灰度预测计算。新增四项使用根目录 `metrics.py` 的 `AS1000Metrics` 与 `evaluate_small_targets.py`：预测阈值严格大于 0.5、8 连通、质心距离严格小于 3 像素的一对一匹配。IoU/F1 汇总全数据集像素计数，Pd 为检出目标数/GT 目标数，Fa 为未匹配预测连通域面积/总图像像素；Fa 不乘 10^6。

完整精度的原四项见服务器 `experiments/SCOD/Seal_original_metrics.json`；新增四项、计数及协议见 [LRDNet.json](小目标指标评估_20261103/LRDNet.json)。新增指标累计 1,983 张图、7,383 个 GT 目标，检出 5 个，未匹配预测面积 5,550,002 像素，总面积 514,338,642 像素。

## 源域训练协议与核对

- 论文：`LRDNet-master/LRDNet(NN25).pdf`。ImageNet-1K 预训练 ResNet50，RGB 输入 384×384，batch size 8，种子 1024，AdamW 与公开代码的三角学习率日程。
- 训练图像为 [LRDNet 作者提供的训练集](https://drive.google.com/file/d/1D9bf1KeeCJsxxri6d2qAC7z6O1X_fxpt/view)中的 4,040 张 S-COD/CAMO 图像，服务器目录为 `data_original/TrainDataset/Imgs/`。
- 使用 [CRNet 作者公开的 Scribble-COD](https://drive.google.com/file/d/1u7PRtZDu2vXCRe0o2SplVYa7ESoZQFR/view)，目录为 `data_original/Scribble/`，与全部训练图像逐名对应。涂鸦编码是 0 未标注、1 前景、2 背景；加载器转为 255 忽略、1 前景、0 背景。`TrainDataset/GT/` 是 0/255 完整掩码，不用作训练标签。
- 第一阶段预先固定训练 100 epoch，取 `experiments/SCOD/stage1/models/100_epoch.pth` 生成 4,040 张训练软伪标签。第二阶段从该权重初始化，软伪标签置信度介于 0.15 与 0.95 的像素忽略，再训练 100 epoch。
- 最终固定权重为 `experiments/SCOD/stage2/models/100_epoch.pth`。两个训练命令均指定 `--validate_epoch 0`，训练日志均完成至第 100 epoch，没有 Seal 验证或选权重记录。已有 Seal 推理日志明确加载该第二阶段权重。
- 不读取 Seal train/val 进行训练、适配、选阈值或选权重。源域测试集只核对实现，不加入训练；Seal test GT 仅用于最终指标计算。

服务器已解压数据、4,040 张软伪标签和两阶段权重均可用；第二阶段检查点包含 972 个状态张量，读取正常且数值有限。现有源域协议符合跨域要求，直接复用其预测与指标。

## 环境、命令与产物

服务器：RSML-3，项目根目录 `/share_datasets/Seal_Model_Reproduction/`；代码目录 `LRDNet-master/`，Python 为 `/home/yqwang/miniforge3/envs/lrdnet_seal/bin/python`。

历史源域训练命令如下，终点在训练前固定。当前已有可用权重，不需要重新执行训练。

```bash
cd /share_datasets/Seal_Model_Reproduction/LRDNet-master
PYTHON=/home/yqwang/miniforge3/envs/lrdnet_seal/bin/python
"$PYTHON" train.py --stage 1 --epochs 100 --batch_size 8 --validate_epoch 0 \
  --experiment_root experiments/SCOD \
  --train_image_root data_original/TrainDataset/Imgs \
  --train_label_root data_original/Scribble
"$PYTHON" generate_softgt.py --stage 1 \
  --checkpoint experiments/SCOD/stage1/models/100_epoch.pth \
  --experiment_root experiments/SCOD \
  --train_image_root data_original/TrainDataset/Imgs
"$PYTHON" train.py --stage 2 --epochs 100 --batch_size 8 --validate_epoch 0 \
  --checkpoint experiments/SCOD/stage1/models/100_epoch.pth \
  --experiment_root experiments/SCOD \
  --train_image_root data_original/TrainDataset/Imgs
```

需要重新生成预测时仅用上述固定第二阶段权重；已有完整预测可直接复用。

```bash
"$PYTHON" infer_seal.py --stage 2 \
  --checkpoint experiments/SCOD/stage2/models/100_epoch.pth \
  --experiment_root experiments/SCOD \
  --data_root /share_datasets/Seal_Model_Reproduction/Seal_data
"$PYTHON" /share_datasets/Seal_Model_Reproduction/evaluate.py \
  --pred_root /share_datasets/Seal_Model_Reproduction/LRDNet-master/experiments/SCOD/test_maps \
  --gt_dir /share_datasets/Seal_Model_Reproduction/Seal_data/test/GT \
  --json_output experiments/SCOD/Seal_original_metrics.json
"$PYTHON" /share_datasets/Seal_Model_Reproduction/evaluate_small_targets.py \
  --pred_root /share_datasets/Seal_Model_Reproduction/LRDNet-master/experiments/SCOD/test_maps \
  --gt_dir /share_datasets/Seal_Model_Reproduction/Seal_data/test/GT \
  --json_output /share_datasets/Seal_Model_Reproduction/docs/小目标指标评估_20261103/LRDNet.json
```

训练日志为 `experiments/SCOD/stage{1,2}_train.log`，伪标签日志为 `generate_softgt.log`；推理和原四项评估日志分别为 `Seal_infer.log`、`Seal_original_evaluate.log`。原四项批量重评估入口为 `reevaluate_scod.sh`。

## 源域核对结果

[作者提供的测试集](https://drive.google.com/file/d/1QEGnP9O7HbN_2tH999O3HRIsErIVYalx/view)实际包含 CAMO-test 250 张、COD10K-test 2,026 张和 CHAMELEON 76 张，不含 README 所称的 NC4K。源域核对复用 CAMO 与 COD10K 的已有预测，不参与权重选择。

| 测试集 | 张数 | S↑ | Fw↑ | E↑ | MAE↓ |
|---|---:|---:|---:|---:|---:|
| CAMO-test | 250 | 0.73615 | 0.63208 | 0.79570 | 0.09293 |
| COD10K-test | 2,026 | 0.73878 | 0.59601 | 0.80972 | 0.04522 |

预测目录为 `experiments/SCOD/{CAMO,COD10K}/test_maps/`；原函数结果及日志为 `experiments/SCOD/{CAMO,COD10K}_original_metrics.json` 和对应的 `_original_evaluate.log`。论文报告的 mean E 与此处连续灰度 E 定义不同，论文 Fβ 与本项目 Fw 也不同，不能直接混作同一指标比较。

## 历史 Seal 框适配（已停止，仅存档）

Seal 框监督训练、Seal val 选权重、排队和自动恢复不再执行。`run_seal.sh`、`run_seal_box_stage2.sh`、`run_seal_box_original_eval.sh`、`audit_existing_candidates.sh` 和 `validate_checkpoint.py` 入口已停用；没有运行中的 LRDNet Seal 适配任务。历史权重、预测、验证图和日志保留，不进入跨域主表。

| 历史实验 | 第一阶段 | 第二阶段 | 历史测试与记录 |
|---|---|---|---|
| `experiments/Seal/` 与 `experiments/Seal_box/` | 原记录 epoch100；仅保留70–100周期权重 | 原记录 epoch80 | `Seal_box/test_maps/`、`Seal_box/evaluate.log`；此前评估口径，不用于主表 |
| `experiments/Seal_box_original_eval/` | 10个验证候选，epoch100，val Fw 0.1085896121；生成1500张新软伪标签 | 10个验证候选，epoch80，val Fw 0.1193291209 | 原四项 S 0.49111、Fw 0.06664、E 0.38246、MAE 0.00259；`Seal_metrics.json` |

旧周期候选按原函数核对的记录保留于 `experiments/Seal_box_original_eval_audit/stage{1,2}_validation.csv`。完整重训历史的每阶段 `validation.csv`、周期权重及每轮500张验证图均保留于对应阶段目录。

## 已知问题

源域训练模型在 Seal 上跨域泛化较弱，低分按实际结果记录。Seal GT 的平均前景面积占比为 0.0206%；现有源域预测在128阈值下有1,042张无前景，像素级 recall 为3.44%。这些诊断不用于重新选择权重、阈值或后处理。
