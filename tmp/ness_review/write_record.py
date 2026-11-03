import json
from pathlib import Path
b=Path('tmp/ness_review')
public=json.loads((b/'published_metrics.json').read_text())
val=json.loads((b/'val_metrics.json').read_text())
test=json.loads((b/'test_metrics.json').read_text())
rows='\n'.join(f"| {name} | 1,983 | {m['S']:.8f} | {m['F']:.8f} | {m['E']:.8f} | {m['MAE']:.8f} |" for name,m in public.items())
candidates='\n'.join(f"| {m['epoch']} | `NESS_general_{m['epoch']}.pth` | 500 | {m['F']:.10f} | {m['MAE']:.10f} |" for m in val)
m=test['test']
doc=f'''# NESS-Net 在 Seal 上的实验记录

## 当前状态与主表协议

35 epoch 框监督训练已完成，7 个周期检查点齐全。四组公开权重预测已按指定原函数重新评估；框适配已复用全部候选的验证预测重新选权重，并完成选定模型的测试。

量化表 NESS-Net 主行固定采用 **公开 `NESS_refinement.pth` + NAMLab 边缘**。这是 RGB-D NESS-Net 使用 Seal 热成像作为第二模态的跨模态迁移测试。主方法不依据 test 分数在权重、阶段或边缘类型之间择优。

## 固定评估口径

四项指标直接调用 `Ours/evaluate.py` 的 `s_measure_folder`、`weighted_f_measure_folder`、`e_measure_folder`、`mae_folder`。本地与服务器根目录 `evaluate.py` 的四函数原文均与该文件一致，函数及内部实现保持固定。E 对连续灰度预测计算。

验证使用 `Seal_data/val/GT`，测试使用 `Seal_data/test/GT`。新进程明确加载服务器 `Ours/evaluate.py`；每组新输出保存 `evaluate_source.py`。所有路径以下均相对服务器项目根目录 `/share_datasets/Seal_Model_Reproduction/`，Python 为 `/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python`。

## 公开权重迁移测试

输入来自原始 `Seal_data/test/RGB` 与 `T`，不用框提示。16 位热图按每图 min/max 线性映射到 8 位，再复制成三通道，送入深度分支。RGB 使用 ImageNet 标准化，第二模态使用 ToTensor；两路输入均为 384×384。

推理取作者返回的第三项 `sal_ref`，双线性恢复到 GT 原尺寸，随后 sigmoid、逐图 min-max 和 round×255 保存灰度 PNG。`general` 权重使用 general 分支，`refinement` 权重使用 refinement 分支。NAMLab 从原尺寸 RGB 和转换后的原尺寸热图生成边缘；Canny 在缩放至 384×384 后使用阈值 10、100。

| 协议 | 预测图 | S↑ | Fw↑ | E↑ | MAE↓ |
|---|---:|---:|---:|---:|---:|
{rows}

四组预测均与全部 1,983 张 test GT 文件名完整对应，直接复用原预测，没有重新推理。权重为 `NESS-Net-main/NESS_{{general,refinement}}.pth`；预测目录为 `results/NESS-Net/Seal/{{general,refinement}}_{{namlab,canny}}_test_maps/`。新结果为 `results/NESS-Net/Seal/original_eval/published_metrics.json`，日志为同目录 `run.log`。

```bash
cd /share_datasets/Seal_Model_Reproduction
/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python -u \\
  NESS-Net-main/reevaluate_published_seal.py
# 单目录统一入口；另外三组替换 pred_root 即可
/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python evaluate.py \\
  --pred_root results/NESS-Net/Seal/refinement_namlab_test_maps \\
  --gt_dir /share_datasets/Seal_Model_Reproduction/Seal_data/test/GT
```

## Seal 框监督、Canny 边缘、general 阶段适配

训练使用 `Seal_data/train/{{RGB,T,box}}` 的 1,500 对输入，从公开 refinement 完整权重严格加载初始化。`--general_stage` 控制模型进入 general 分支，训练没有 teacher、伪标签再生成或 refinement 第二阶段。测试对应 `--stage general`，不因公开初始化权重名包含 refinement 而切换分支。

配置为输入 384×384、batch size 2、Adam 初始学习率 1e-5、35 epoch，按作者调度每 20 epoch 乘 0.9，每 5 epoch 保存独立权重。原训练日志 `NESS-Net-main/experiments/Seal_box/canny_train.log` 记录完成第 35 epoch 的 750/750 step。

### 输入和监督核对

- RGB 训练链接逐一指向原始 train RGB；热图转换逐像素匹配原始 train T 的每图线性归一化结果。
- 每个框先裁剪到有效图像坐标。框中心区域为前景，框外为背景，框内其余部分未标注。`gt` 前景为 255、其余为 0；`mask` 已标注为 255、未标注为 0。加载后除以 255。
- 标签与 mask 沿用作者的双线性缩放，边界为软标签和软 mask；损失沿用作者的 masked BCE、IOU、平滑及边缘监督。该实际缩放协议保留在现有训练产物中。
- 训练预计算 Canny 与推理即时 Canny 均对 384×384 RGB/热图使用 10、100 阈值。1,500 对输入的热图、Canny、标签和 mask 与准备规则全部一致，未发现需要补训的输入实现偏差。
- `val/GT` 仅用于选权重，`test/GT` 仅用于最终评估。模型输入不使用 `Ours/vis.zip` 的标框可视化。

### 原 Fw 验证重选

候选权重均在 `NESS-Net-main/experiments/Seal_box/canny_checkpoints/`。旧 `canny_selected/val_epoch_{{epoch}}_maps/` 的 7 组预测各有 500 张，文件名、尺寸与 val GT 全部对应，生成日志和旧 selector 的 checkpoint/epoch 对应关系明确，因此复用预测。

| epoch | 候选权重 | val 预测 | 原 Fw↑ | 原 MAE↓ |
|---:|---|---:|---:|---:|
{candidates}

按 Fw 降序、MAE 升序选定 **第 {test['val']['epoch']} epoch**，完整权重路径为 `{test['checkpoint']}`。新记录目录为 `NESS-Net-main/experiments/Seal_box/canny_selected_original_eval/`：`val_metrics.json` 保存每个候选路径、预测路径及验证指标，`selection.json` 保存选择规则和最终权重。

### 选定权重测试

按原 Fw 选定后重新推理完整 test，得到 1,983 张预测。适配结果为：

| 协议 | S↑ | Fw↑ | E↑ | MAE↓ |
|---|---:|---:|---:|---:|
| Seal 框监督 / Canny / general / epoch {test['val']['epoch']} | {m['S']:.8f} | {m['F']:.8f} | {m['E']:.8f} | {m['MAE']:.8f} |

预测位于新记录目录 `test_maps/`，指标为 `test_metrics.json`，推理日志为 `test_maps_infer.log`，总日志为 `run.log`。此次复用了训练权重及完整验证预测，没有重新训练；测试重新推理对应原 Fw 重选后的权重。旧目录及预测保留。

```bash
cd /share_datasets/Seal_Model_Reproduction/NESS-Net-main
CUDA_VISIBLE_DEVICES=0 /home/yqwang/miniforge3/envs/fct_sam_seal/bin/python -u select_seal_box.py \\
  --seal-dir /share_datasets/Seal_Model_Reproduction/Seal_data \\
  --checkpoints experiments/Seal_box/canny_checkpoints \\
  --reuse-dir experiments/Seal_box/canny_selected \\
  --output-dir experiments/Seal_box/canny_selected_original_eval
```

## 已知问题

- 热图替代深度属于跨模态迁移；Canny 边缘、Seal 框监督和单阶段适配均为实际实验设置，不等同于论文完整两阶段涂鸦监督协议。
- 热图 3000、3228 的 NAMLab 输出没有层级图，3250、3290 仅有一个均匀区域，四张边缘为空。独立重新运行 NAMLab 得到与已有边缘逐像素相同的结果，未添加替代边缘。核对产物位于 `results/NESS-Net/Seal/original_eval/empty_thermal_{{inputs,edges}}/`。
- 作者 refinement 分支的 `F.dropout(..., p=0.25)` 在 eval 时仍生效。公开预测沿用作者测试实现及固定随机种子，未改变其分支行为；general 阶段框适配不经过该 dropout 分支。
- 服务器 NESS-Net 目录及项目内未发现可用的 NJU2K、NLPR 或 LFSD 源域测试集，本记录不提供源域性能核对。

## 历史评估记录

下表采用此前评估实现，E 为此前阈值评估口径，仅保留历史来源，不用于最终比较。

| 公开协议 | S↑ | Fw↑ | E↑ | MAE↓ |
|---|---:|---:|---:|---:|
| general + NAMLab | 0.44919 | 0.00076 | 0.27390 | 0.08516 |
| refinement + NAMLab | 0.46706 | 0.00171 | 0.30600 | 0.04230 |
| general + Canny | 0.45475 | 0.00077 | 0.28260 | 0.07416 |
| refinement + Canny | 0.47068 | 0.00173 | 0.32446 | 0.03665 |

旧适配选择采用第 15 epoch，旧验证 S 0.67989715、Fw 0.46954164、E 0.79198087、MAE 0.0001258321；旧测试 S 0.71380369、Fw 0.46897830、E 0.87039130、MAE 0.0002239284。来源为 `NESS-Net-main/experiments/Seal_box/canny_selected/{{val_metrics,test_metrics}}.json`。
'''
Path('docs/NESS-Net_Seal实验记录.md').write_text(doc,encoding='utf-8')
