# MIR-Net：Seal 跨域测试与框监督适配

## 跨域主表协议

主表采用作者公开完整权重在 Seal 上的跨域测试。Seal 框稀疏标签和 SAM 框伪标签训练仅作历史适配实验，不再执行或恢复，不使用 Seal 验证集选择主表权重。2026-11-03 已停止仍在运行的 `Seal_box_sam_150_original_eval` 训练及其启动脚本和子进程；已有产物保留。本文适配部分的训练配置与进度为历史记录。

## 固定评估协议

`Ours/evaluate.py` 的 `s_measure_folder`、`weighted_f_measure_folder`、`e_measure_folder`、`mae_folder` 是唯一指标实现。本地根目录和服务器根目录 `evaluate.py` 的四个函数均已逐函数比较源码并确认一致。四个函数及内部实现未修改。E 直接使用原函数对连续灰度预测计算。

RGB 输入 384×384，前景与背景 logits 相减后 sigmoid、逐图 min-max，再按 GT 原尺寸保存 8 位灰度 PNG。验证与测试使用相同后处理。test GT 显式指定为 `/share_datasets/Seal_Model_Reproduction/Seal_data/test/GT`，每项最终预测均核对 1,983 个文件名与 GT 完整对应。

## 公开权重跨域测试

- 权重：`MIR-Net-main/published_checkpoint/31_wM_test1/model-best.pth`，PVTv2-B4。作者归档日志的最后一次最佳记录为第 133 epoch，MAE 0.0501738837；权重文件自身不含 epoch 元数据。Seal 不参与公开权重选择。
- 保留已有 `results/MIR-Net/Seal/published_test_maps/` 的完整 1,983 张预测，未重新训练或推理，直接按固定原函数重评估。

| 实验 | 选定 epoch | S↑ | Fw↑ | E↑ | MAE↓ |
|---|---:|---:|---:|---:|---:|
| 公开权重跨域测试 | 作者日志 133 | 0.47532 | 0.00441 | 0.29026 | 0.01843 |

```bash
cd /share_datasets/Seal_Model_Reproduction/MIR-Net-main
/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python audit_fixed_evaluation.py \
  --root /share_datasets/Seal_Model_Reproduction --reference original_evaluate.py \
  --pred /share_datasets/Seal_Model_Reproduction/results/MIR-Net/Seal/published_test_maps \
  --gt_dir /share_datasets/Seal_Model_Reproduction/Seal_data/test/GT --output published_original_metrics.json
```

## Seal 适配训练

以下三项是 Seal 适配历史产物，不进入跨域主表。三项均从公开完整权重初始化。RGB 384×384，batch size 2、梯度累积 4，AdamW，初始学习率 1e-5、weight decay 1e-4。每 5 epoch 对完整 500 张 val 计算原始 Fw，并保存该轮检查点、验证预测和记录。`model-best.pth` 按原函数 val Fw 更新；test 不参与选择。

| 实验 | 标签来源 | 交叉熵输入 | epoch | 学习率日程 |
|---|---|---|---:|---|
| 框稀疏标签 | 复用 `experiments/Seal/box_labels/` | sigmoid 概率 | 60 | 固定 1e-5 |
| SAM 框伪标签 | 复用 `experiments/Seal_box_sam/labels/` | 原始 logits | 60 | 固定 1e-5 |
| SAM 框伪标签 | 同上 | 原始 logits | 150 | 每 50 epoch ×0.1 |

两套标签均有 1,500 张。训练监督仅来自 train RGB 与 box，未读取 train GT 生成监督。稀疏标签与 SAM 的损失输入协议分别保留；此次工作没有将损失修正混入评估重算。SAM 适配单独标为 MIR-Net + SAM 框提示伪标签。

旧实验三组都只保存了历史最佳 `model-best.pth`，`val_maps/` 覆盖至最后一次验证，无法恢复完整候选。因此三组均在独立 `_original_eval` 目录重新训练，旧训练与测试产物全部保留。150 epoch 旧实验已完成，旧日志最佳为第 125 epoch；其训练和测试结束于服务器根评估文件恢复之前。运行中已导入的旧评估模块不能靠替换磁盘文件更新。

历史重训启动时保存并加载 `evaluate_source.py` 副本，保存 `config.json`、训练/推理/损失/网络源码。各验证轮次保留 `model-epoch-NNN.pth`、`val_maps/epoch-NNN/`；`validation.jsonl` 记录 epoch、权重路径、val Fw、学习率，`selection.json` 记录最终选择。

### 框稀疏标签，60 epoch

- 重训入口：`bash MIR-Net-main/run_finetune_seal.sh 1`；设备可按实际空闲情况指定 0 或 1。
- 权重和验证产物：`MIR-Net-main/experiments/Seal_original_eval/`。
- 测试预测：`results/MIR-Net/Seal/Seal_original_eval_test_maps/`。

状态：训练到第 60 epoch 后停止；未生成该轮选定权重的完整 test 预测，已有检查点、验证图和日志保留。

### SAM 框提示伪标签，60 epoch

- 重训入口：`bash MIR-Net-main/run_box_sam_seal.sh 1`；设备可按实际空闲情况指定 0 或 1。
- 权重和验证产物：`MIR-Net-main/experiments/Seal_box_sam_original_eval/`。
- 测试预测：`results/MIR-Net/Seal/Seal_box_sam_original_eval_test_maps/`。

状态：训练到第 60 epoch 后停止；未生成该轮选定权重的完整 test 预测，已有检查点、验证图和日志保留。

### SAM 框提示伪标签，150 epoch

- 重训入口：`bash MIR-Net-main/run_box_sam_seal_150.sh 1`；设备可按实际空闲情况指定 0 或 1。
- 权重和验证产物：`MIR-Net-main/experiments/Seal_box_sam_150_original_eval/`。
- 测试预测：`results/MIR-Net/Seal/Seal_box_sam_150_original_eval_test_maps/`。

状态：训练到第 94 epoch 后停止；未生成完整 150 epoch 协议的 test 预测，已有检查点、验证图和日志保留。

## 旧权重的原函数重评估

此表只重算旧预测。权重仍由旧验证口径选择，不能声明为原函数 val Fw 下的全程最佳，也不代替上面的独立重训结果。

| 旧实验 | 旧口径选定 epoch | S↑ | Fw↑ | E↑ | MAE↓ |
|---|---:|---:|---:|---:|---:|
| 框稀疏标签 60 epoch | 45 | 0.48498 | 0.03627 | 0.29639 | 0.00518 |
| SAM 框伪标签 60 epoch | 60 | 0.48221 | 0.02630 | 0.27956 | 0.00787 |
| SAM 框伪标签 150 epoch | 125 | 0.48866 | 0.04485 | 0.31645 | 0.00382 |

四组旧预测均为 1,983 张，重评估明细见 `docs/MIR-Net_原函数评估_20261103/legacy_reassessment.json`。历史实验的配置与验证记录保留在服务器 `MIR-Net-main/experiments/`；大权重、训练标签和预测图保留在服务器。

## 命令与产物

已完成的跨域推理复用 `published_test_maps/`。历史适配评估入口定义如下，适配任务已停止：

```bash
/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python audit_fixed_evaluation.py \
  --root /share_datasets/Seal_Model_Reproduction \
  --reference experiments/EXPERIMENT_original_eval/evaluate_source.py \
  --pred /share_datasets/Seal_Model_Reproduction/results/MIR-Net/Seal/EXPERIMENT_original_eval_test_maps \
  --gt_dir /share_datasets/Seal_Model_Reproduction/Seal_data/test/GT \
  --output experiments/EXPERIMENT_original_eval/test_metrics.json
```

## 小目标指标（同批预测）

2026-11-03，复用本表协议下原四项指标对应的 1,983 张测试预测，使用统一 `AS1000Metrics` 计算。预测目录：`/share_datasets/Seal_Model_Reproduction/results/MIR-Net/Seal/published_test_maps`。

| IoU↑ | F1↑ | Pd↑ | Fa↓（原始比例） |
|---:|---:|---:|---:|
| 0.00080272 | 0.00160416 | 0.00243803 | 0.0182267484 |

完整精度、计数及协议见[MIR-Net.json](小目标指标评估_20261103/MIR-Net.json)；横向结果见[Seal 小目标指标评估](Seal小目标指标评估汇总.md)。
