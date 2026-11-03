# NESS-Net 在 Seal 上的实验记录

## 当前跨域主表协议

主表采用作者公开完整权重 **`NESS_refinement.pth` + NAMLab 边缘 + refinement 分支**，属于源域涂鸦监督 RGB-D 模型以 Seal 热成像替代深度的跨模态迁移测试。权重与边缘流程预先固定，不使用 Seal train/val 训练、选择权重、阈值或调参。

NESS-Net 的 Seal 适配训练、标签生成和自动跟进均停止执行。服务器未发现运行或等待启动的相关进程、crontab、at 或用户定时器，本地未发现 NESS-Net Codex 自动化或 Windows 计划任务。旧 `select_seal_box_after_train.sh` 已停用。历史 35 epoch 框适配及 epoch5 的全部结果保留在本文历史章节。

| 主表产物 | 路径 |
|---|---|
| 作者公开完整权重 | `/share_datasets/Seal_Model_Reproduction/NESS-Net-main/NESS_refinement.pth` |
| 八项指标共同预测 | `/share_datasets/Seal_Model_Reproduction/results/NESS-Net/Seal/refinement_namlab_test_maps/`（1,983 张） |
| 测试 GT | `/share_datasets/Seal_Model_Reproduction/Seal_data/test/GT/` |
| 原四项完整精度 | [published_metrics.json](NESS-Net_Seal评估产物/published_metrics.json) 的 `refinement_namlab` |
| 小目标计数和协议 | [refinement_namlab_small_target_metrics.json](NESS-Net_Seal评估产物/refinement_namlab_small_target_metrics.json) |
| 统一八项与协议 | [cross_domain_metrics.json](NESS-Net_Seal评估产物/cross_domain_metrics.json) |

## 跨域主表八项结果

| 指标 | 原始比例 |
|---|---:|
| S | 0.4666185279426068 |
| Fw | 0.001733883532895621 |
| E | 0.263767112039326 |
| MAE | 0.042296525090932846 |
| IoU | 0.0007333271181027003 |
| F1 | 0.001465579487023831 |
| Pd | 0.005146959230665041 |
| Fa | 0.032954249624511 |

原四项从完整精度 JSON 读取；新增四项直接评估同一批公开 refinement + NAMLab 预测，没有使用 Seal 框适配的预测或指标。Fa 保存原始比例。主表 NESS-Net 仅一行，结果区仅第 1～20 行。

小目标评估共 1,983 张、514,338,642 像素，TP/FP/FN 为 12,554 / 17,013,524 / 93,158；GT 组件 7,383，预测组件 8,410，成功匹配 38，未匹配预测组件面积 16,949,644 像素。评估入口核对了全部 PNG 文件配对与原尺寸一致性。

## 固定评估与输入

S、Fw、E、MAE 使用 `Ours/evaluate.py` 的四个原函数，服务器根目录四函数原文与其一致，未修改内部实现。E 对连续灰度预测计算。新增指标使用根目录 `metrics.py` 的 `AS1000Metrics` 和 `evaluate_small_targets.py`：预测 >0.5，GT >128，8 连通，质心距离严格小于 3 像素的栅格顺序一对一匹配。IoU/F1 为全数据集像素累计，Pd 为检出组件/GT 组件，Fa 为未匹配预测组件面积/总像素数。

16 位热成像按每图 `lo=min(T)`、`hi=max(T)` 映射为 `round((T-lo)*255/(hi-lo))` 的 uint8，再复制三通道；恒定热图映射为零。RGB 使用 ImageNet 标准化，热图仅 ToTensor，输入均为 384×384。NAMLab 对原尺寸 RGB 与转换后的原尺寸热图生成边缘，读取后缩放至 384×384。推理保留作者 `sal_ref` 输出分支、双线性恢复尺寸、sigmoid、逐图 min-max 和 round×255 保存流程。

服务器项目根目录为 `/share_datasets/Seal_Model_Reproduction/`，Python 为 `/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python`。新增结果位于 `results/NESS-Net/Seal/original_eval/refinement_namlab_small_target_metrics.json`，日志为同目录 `refinement_namlab_small_target_evaluate.log`。原公开预测完整且来源明确，直接复用；没有重新训练或推理。

```bash
cd /share_datasets/Seal_Model_Reproduction
/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python evaluate_small_targets.py \
  --pred_root results/NESS-Net/Seal/refinement_namlab_test_maps \
  --gt_dir /share_datasets/Seal_Model_Reproduction/Seal_data/test/GT \
  --json_output results/NESS-Net/Seal/original_eval/refinement_namlab_small_target_metrics.json
```

## 源域数据归档

作者公开完整权重在服务器可用，采用该权重完成跨域测试。本地 `data/NESS-Net/train_data.zip` 中 `img`、`depth`、`gray`、`gt`、`gt_mask`、`mask`、`gt_refinement`、`rgb-lab`、`depth-lab` 九组目录均有 2,185 个文件；`test_data.zip` 的五组 `img/depth/gt/rgb-lab/depth-lab` 每组 4,579 个文件，按 STERE、ReDWeb-S、SIP、NJU2K_Test、DUTLF-Depth、NLPR_Test、DES、RGBD135、LFSD、SSD 分目录保存。服务器未发现这些归档或已解压源数据。源域训练仅在公开完整权重不可用时开展，使用原有非 Seal 数据及监督形式。

## 公开权重迁移测试

输入来自原始 `Seal_data/test/RGB` 与 `T`，不用框提示。16 位热图按每图 min/max 线性映射到 8 位，再复制成三通道，送入深度分支。RGB 使用 ImageNet 标准化，第二模态使用 ToTensor；两路输入均为 384×384。

推理取作者返回的第三项 `sal_ref`，双线性恢复到 GT 原尺寸，随后 sigmoid、逐图 min-max 和 round×255 保存灰度 PNG。`general` 权重使用 general 分支，`refinement` 权重使用 refinement 分支。NAMLab 从原尺寸 RGB 和转换后的原尺寸热图生成边缘；Canny 在缩放至 384×384 后使用阈值 10、100。

| 协议 | 预测图 | S↑ | Fw↑ | E↑ | MAE↓ |
|---|---:|---:|---:|---:|---:|
| refinement_namlab | 1,983 | 0.46661853 | 0.00173388 | 0.26376711 | 0.04229653 |
| general_namlab | 1,983 | 0.44908160 | 0.00076776 | 0.25489362 | 0.08516204 |
| general_canny | 1,983 | 0.45462905 | 0.00077448 | 0.25593078 | 0.07415722 |
| refinement_canny | 1,983 | 0.47015079 | 0.00176020 | 0.26829350 | 0.03664672 |

四组预测均与全部 1,983 张 test GT 文件名完整对应，直接复用原预测，没有重新推理。权重为 `NESS-Net-main/NESS_{general,refinement}.pth`；预测目录为 `results/NESS-Net/Seal/{general,refinement}_{namlab,canny}_test_maps/`。新结果为 `results/NESS-Net/Seal/original_eval/published_metrics.json`，日志为同目录 `run.log`。

```bash
cd /share_datasets/Seal_Model_Reproduction
/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python -u \
  NESS-Net-main/reevaluate_published_seal.py
# 单目录统一入口；另外三组替换 pred_root 即可
/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python evaluate.py \
  --pred_root results/NESS-Net/Seal/refinement_namlab_test_maps \
  --gt_dir /share_datasets/Seal_Model_Reproduction/Seal_data/test/GT
```

## 历史 Seal 框监督适配（不用于跨域主表）

训练使用 `Seal_data/train/{RGB,T,box}` 的 1,500 对输入，从公开 refinement 完整权重严格加载初始化。`--general_stage` 控制模型进入 general 分支，训练没有 teacher、伪标签再生成或 refinement 第二阶段。测试对应 `--stage general`，不因公开初始化权重名包含 refinement 而切换分支。

配置为输入 384×384、batch size 2、Adam 初始学习率 1e-5、35 epoch，按作者调度每 20 epoch 乘 0.9，每 5 epoch 保存独立权重。原训练日志 `NESS-Net-main/experiments/Seal_box/canny_train.log` 记录完成第 35 epoch 的 750/750 step。

### 输入和监督核对

- RGB 训练链接逐一指向原始 train RGB；热图转换逐像素匹配原始 train T 的每图线性归一化结果。
- 每个框先裁剪到有效图像坐标。框中心区域为前景，框外为背景，框内其余部分未标注。`gt` 前景为 255、其余为 0；`mask` 已标注为 255、未标注为 0。加载后除以 255。
- 标签与 mask 沿用作者的双线性缩放，边界为软标签和软 mask；损失沿用作者的 masked BCE、IOU、平滑及边缘监督。该实际缩放协议保留在现有训练产物中。
- 训练预计算 Canny 与推理即时 Canny 均对 384×384 RGB/热图使用 10、100 阈值。1,500 对输入的热图、Canny、标签和 mask 与准备规则全部一致，未发现需要补训的输入实现偏差。
- `val/GT` 仅用于选权重，`test/GT` 仅用于最终评估。模型输入不使用 `Ours/vis.zip` 的标框可视化。

### 原 Fw 验证重选

候选权重均在 `NESS-Net-main/experiments/Seal_box/canny_checkpoints/`。旧 `canny_selected/val_epoch_{epoch}_maps/` 的 7 组预测各有 500 张，文件名、尺寸与 val GT 全部对应，生成日志和旧 selector 的 checkpoint/epoch 对应关系明确，因此复用预测。

| epoch | 候选权重 | val 预测 | 原 Fw↑ | 原 MAE↓ |
|---:|---|---:|---:|---:|
| 5 | `NESS_general_5.pth` | 500 | 0.4679960478 | 0.0001338962 |
| 10 | `NESS_general_10.pth` | 500 | 0.4626077951 | 0.0001276297 |
| 15 | `NESS_general_15.pth` | 500 | 0.4665612005 | 0.0001258321 |
| 20 | `NESS_general_20.pth` | 500 | 0.4404877771 | 0.0001264964 |
| 25 | `NESS_general_25.pth` | 500 | 0.4517579676 | 0.0001234448 |
| 30 | `NESS_general_30.pth` | 500 | 0.4529817784 | 0.0001225499 |
| 35 | `NESS_general_35.pth` | 500 | 0.4184263070 | 0.0001234451 |

按 Fw 降序、MAE 升序选定 **第 5 epoch**，完整权重路径为 `/share_datasets/Seal_Model_Reproduction/NESS-Net-main/experiments/Seal_box/canny_checkpoints/NESS_general_5.pth`。新记录目录为 `NESS-Net-main/experiments/Seal_box/canny_selected_original_eval/`：`val_metrics.json` 保存每个候选路径、预测路径及验证指标，`selection.json` 保存选择规则和最终权重。

### 选定权重测试

按原 Fw 选定后对完整 test 推理，得到 1,983 张预测。Seal 框监督、T 替代深度、Canny、general epoch5 的统一八项结果为：

| 指标 | 值 |
|---|---:|
| S↑ | 0.49832819838003467 |
| Fw↑ | 0.48279346633848436 |
| E↑ | 0.6618007353805426 |
| MAE↓ | 0.0002622979518491775 |
| IoU↑ | 0.2515188538720313 |
| F1↑ | 0.4019417735399922 |
| Pd↑ | 0.567384532033049 |
| Fa↓ | 0.00007798169673590265 |

预测位于新记录目录 `test_maps/`，指标为 `test_metrics.json`，推理日志为 `test_maps_infer.log`，总日志为 `run.log`。此次复用了训练权重及完整验证预测，没有重新训练；测试重新推理对应原 Fw 重选后的权重。旧目录及预测保留。

```bash
cd /share_datasets/Seal_Model_Reproduction/NESS-Net-main
CUDA_VISIBLE_DEVICES=0 /home/yqwang/miniforge3/envs/fct_sam_seal/bin/python -u select_seal_box.py \
  --seal-dir /share_datasets/Seal_Model_Reproduction/Seal_data \
  --checkpoints experiments/Seal_box/canny_checkpoints \
  --reuse-dir experiments/Seal_box/canny_selected \
  --output-dir experiments/Seal_box/canny_selected_original_eval
```

### 历史框适配小目标指标

使用上述框适配第 5 epoch 的同一批测试预测，通过项目公共 `AS1000Metrics` 评估；结果并入上面的八项指标表。定义与文献代码对照见 [小目标评价指标_定义与代码对照.md](小目标评价指标_定义与代码对照.md)。

结果为 0～1 原始比例。预测阈值为 >0.5，8 连通域，质心距离严格小于 3 像素、一对一匹配；IoU、F1 使用全数据集累计像素，Pd 使用累计目标计数，Fa 为未匹配预测组件面积除以总像素数。

共评估 1,983 张原尺寸图像、514,338,642 个像素。像素 TP/FP/FN 为 45,084 / 73,535 / 60,628；GT 组件 7,383 个、预测组件 6,019 个，匹配成功 4,189 个，未匹配预测组件面积为 40,109 像素。Fa 在汇总表中保存原始比例。

完整结果及参数见 [small_target_metrics.json](NESS-Net_Seal评估产物/small_target_metrics.json)，服务器结果位于 `NESS-Net-main/experiments/Seal_box/canny_selected_original_eval/small_target_metrics.json`。

```bash
cd /share_datasets/Seal_Model_Reproduction
/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python evaluate_small_targets.py \
  --pred_root NESS-Net-main/experiments/Seal_box/canny_selected_original_eval/test_maps \
  --gt_dir /share_datasets/Seal_Model_Reproduction/Seal_data/test/GT \
  --json_output NESS-Net-main/experiments/Seal_box/canny_selected_original_eval/small_target_metrics.json
```

## 已知问题


- 热图替代深度属于跨模态迁移；Canny 边缘、Seal 框监督和单阶段适配均为实际实验设置，不等同于论文完整两阶段涂鸦监督协议。
- 热图 3000、3228 的 NAMLab 输出没有层级图，3250、3290 仅有一个均匀区域，四张边缘为空。独立重新运行 NAMLab 得到与已有边缘逐像素相同的结果，未添加替代边缘。核对产物位于 `results/NESS-Net/Seal/original_eval/empty_thermal_{inputs,edges}/`。
- 作者 refinement 分支的 `F.dropout(..., p=0.25)` 在 eval 时仍生效。公开预测沿用作者测试实现及固定随机种子，未改变其分支行为；general 阶段框适配不经过该 dropout 分支。
- 现有作者公开完整权重可用；本记录尚未提供源域测试集性能核对。

## 历史评估记录

下表采用此前评估实现，E 为此前阈值评估口径，仅保留历史来源，不用于最终比较。

| 公开协议 | S↑ | Fw↑ | E↑ | MAE↓ |
|---|---:|---:|---:|---:|
| general + NAMLab | 0.44919 | 0.00076 | 0.27390 | 0.08516 |
| refinement + NAMLab | 0.46706 | 0.00171 | 0.30600 | 0.04230 |
| general + Canny | 0.45475 | 0.00077 | 0.28260 | 0.07416 |
| refinement + Canny | 0.47068 | 0.00173 | 0.32446 | 0.03665 |

旧适配选择采用第 15 epoch，旧验证 S 0.67989715、Fw 0.46954164、E 0.79198087、MAE 0.0001258321；旧测试 S 0.71380369、Fw 0.46897830、E 0.87039130、MAE 0.0002239284。来源为 `NESS-Net-main/experiments/Seal_box/canny_selected/{val_metrics,test_metrics}.json`。
