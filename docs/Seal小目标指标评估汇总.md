# Seal 小目标指标评估

评估日期：2026-11-03。六组结果均使用量化主表中 S、Fw、E、MAE 对应的同一批 1,983 张预测；不重新选择权重，不重新推理。NESS-Net 复用已完成的同目录小目标评估结果。**Ours 行于 2026-11-05 改用 1105 批次预测重算**（主表第 20 行同步），其余五组仍为 2026-11-03 结果。

统一入口为根目录 `evaluate_small_targets.py`，指标实现为 `metrics.py` 的 `AS1000Metrics`。定义见[小目标评价指标：定义与代码对照](小目标评价指标_定义与代码对照.md)。固定预测 >0.5、GT >128、8 连通、质心距离严格小于 3 像素的扫描顺序一对一匹配。IoU/F1 为累计像素计数；Pd 为检出目标数/GT 目标数；Fa 为未匹配预测连通域面积/总像素数。所有表值都是原始比例。

## 结果

| 方法 | IoU↑ | F1↑ | Pd↑ | Fa↓（原始比例） |
|---|---:|---:|---:|---:|
| FCT-SAM-WSCOD | 0.00067187 | 0.00134283 | 0.00257348 | 0.0346740446 |
| LRDNet | 0.00064195 | 0.00128307 | 0.00067723 | 0.0107905600 |
| MIR-Net | 0.00080272 | 0.00160416 | 0.00243803 | 0.0182267484 |
| SSFam | 0.00099438 | 0.00198679 | 0.28741704 | 0.0402274131 |
| NESS-Net | 0.00073333 | 0.00146558 | 0.00514696 | 0.0329542496 |
| Ours | 0.51930934 | 0.68361238 | 0.89150752 | 0.0000171735 |

每组 GT 共 7,383 个连通域、514,338,642 个像素。PNG 文件名及原尺寸须完全对应；不额外进行归一化、形态学处理或面积筛选。原四项指标及其固定函数保持不变。

NESS-Net 行采用量化主表同一行的公开 refinement + NAMLab 跨域预测。Seal 框监督适配第 5 epoch 的历史小目标结果为 IoU 0.25151885、F1 0.40194177、Pd 0.56738453、Fa 0.0000779817，仅存档于 [NESS-Net_Seal框适配_epoch5.json](小目标指标评估_20261103/NESS-Net_Seal框适配_epoch5.json) 与 [NESS-Net_Seal实验记录.md](NESS-Net_Seal实验记录.md)，不进入主表。

## 预测及原四项指标来源

服务器根目录为 `/share_datasets/Seal_Model_Reproduction/`。JSON 保留完整精度、计数、预测目录和协议。

### FCT-SAM-WSCOD

公开检查点跨域测试；RGB/Scribble。

- 预测：`/share_datasets/Seal_Model_Reproduction/results/FCT-SAM-WSCOD/Seal/test_maps`。
- 原四项指标记录：`/share_datasets/Seal_Model_Reproduction/results/FCT-SAM-WSCOD/original_functions_20261103/cross_domain_metrics_full.json`。
- 小目标完整结果：[FCT-SAM-WSCOD.json](小目标指标评估_20261103/FCT-SAM-WSCOD.json)。

### LRDNet

S-COD原始涂鸦训练；stage2 epoch100；Seal跨域测试。

- 预测：`/share_datasets/Seal_Model_Reproduction/LRDNet-master/experiments/SCOD/test_maps`。
- 原四项指标记录：`/share_datasets/Seal_Model_Reproduction/LRDNet-master/experiments/SCOD/Seal_original_evaluate.log`。
- 小目标完整结果：[LRDNet.json](小目标指标评估_20261103/LRDNet.json)。

### MIR-Net

作者公开完整权重跨域测试；RGB/Scribble。

- 预测：`/share_datasets/Seal_Model_Reproduction/results/MIR-Net/Seal/published_test_maps`。
- 原四项指标记录：`/share_datasets/Seal_Model_Reproduction/docs/MIR-Net_原函数评估_20261103/legacy_reassessment.json`。
- 小目标完整结果：[MIR-Net.json](小目标指标评估_20261103/MIR-Net.json)。

### SSFam

作者公开RGBT完整权重跨域测试。

- 预测：`/share_datasets/Seal_Model_Reproduction/results/SSFam/Seal/RGBT_published`。
- 原四项指标记录：`/share_datasets/Seal_Model_Reproduction/SSFam-main/experiments/Published_fixed_eval/metrics.json`。
- 小目标完整结果：[SSFam.json](小目标指标评估_20261103/SSFam.json)。

### NESS-Net

作者公开refinement完整权重+NAMLab边缘的跨域测试；源域Scribble；T替代深度。

- 预测：`/share_datasets/Seal_Model_Reproduction/results/NESS-Net/Seal/refinement_namlab_test_maps`。
- 原四项指标记录：`/share_datasets/Seal_Model_Reproduction/results/NESS-Net/Seal/original_eval/published_metrics.json` 的 `refinement_namlab`。
- 小目标完整结果：[NESS-Net.json](小目标指标评估_20261103/NESS-Net.json)（与 [refinement_namlab_small_target_metrics.json](NESS-Net_Seal评估产物/refinement_namlab_small_target_metrics.json) 同源）。

### Ours

已有测试预测；RGB-T/Box。

- 预测：`/share_datasets/Seal_Model_Reproduction/Ours/test_maps`（1105 批次）。
- 原四项指标记录：服务器 `results/ours1105_metrics.json`，本地副本 `docs/Ours_Seal评估产物/ours1105_metrics.json`。
- 小目标完整结果：服务器 `results/ours1105_small.json`，本地副本 [ours1105_small.json](Ours_Seal评估产物/ours1105_small.json)；旧预测结果见 [Ours.json](小目标指标评估_20261103/Ours.json)。

## 复算

```bash
cd /share_datasets/Seal_Model_Reproduction
/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python evaluate_small_targets.py \
  --pred_root <上述预测目录> \
  --gt_dir /share_datasets/Seal_Model_Reproduction/Seal_data/test/GT \
  --json_output <结果JSON路径>
```

## 核对结果

1. **计数自洽**：六个 JSON 的 IoU、F1、Pd、Fa 均可由存储的 `tp/fp/fn/n_detected/n_gt/false_alarm_pixels/total_pixels` 精确复算，差值 0（复算脚本 `tmp/verify_small_target_jsons.py`）。`n_gt = 7,383`、`total_pixels = 514,338,642` 在六组间完全一致，即同一批 GT 与同一批 1,983 张原尺寸预测。
2. **像素数与独立记录一致**：`total_pixels / 1,983 = 259,374 = 417×622`，与 Ours 记录的 GT 尺寸、GT 前景占比 0.00020553（tp+fn = 105,712）一致；Ours 1105 批次的像素 precision/recall 58.13%/82.96% 也可由 tp+fp、tp+fn 复现（旧预测为 64.64%/59.39%）。
3. **协议一致**：六组均为 `AS1000Metrics` 默认参数（阈值 >0.5、GT >128、8 连通、距离 <3、全数据集累计），`protocol` 字段逐项相同。
4. **与 `docs/小目标评价指标.pdf` 草稿的差异**：该草稿使用 `pred > 0`、像素级假阳 Fa、`距离 <= 3` 与 ×100 百分数；本项目按已发表官方代码采用 `prob > 0.5`、未匹配连通域 Fa、`距离 < 3` 和 0～1 原始比例。差异明细与数值影响（Ours 旧预测的 Fa 相差 31.8 倍、1105 批次为 7.15 倍）见[小目标评价指标：定义与代码对照](小目标评价指标_定义与代码对照.md#与-docs小目标评价指标pdf-参考实现的差异)。表值只与同口径结果比较。
5. **跨方法可比性**：五个对比方法的 `*/infer_seal.py` 都对预测做逐图 min-max 归一化，>0.5 因此是各图动态范围的中点而非校准概率，弱响应图会被拉伸。SSFam 的 Pd 0.28742 与 IoU 0.00099 并存即为典型：大量斑点落在 GT 质心 3 像素内，但几乎没有像素重叠。Pd/Fa 须与 IoU、F1 合看，见定义对照的已知问题。

主表行值可由 `python tmp/apply_ness_cross_domain_row.py` 写回 `docs/Seal量化SOTA表.xlsx`（需先关闭 Excel）；本次核对已生成同内容副本 `tmp/backup/Seal量化SOTA表.with_ness_cross_domain.xlsx`。

## 全监督 IRSTD 基线（Seal 训练 → Seal 测试）

新增的三个全监督基线**不是**跨域结果：训练用 Seal train 全像素 GT，推理在 Seal test，属于同域全监督，单独列出，不与上表混填。协议、配置与四项指标见[全监督基线汇总](Seal全监督基线评估汇总.md)。

| 方法 | IoU↑ | F1↑ | Pd↑ | Fa↓（原始比例） |
|---|---:|---:|---:|---:|
| DNA-Net（TIP22，val 第 120 epoch） | 0.30000332 | 0.46154239 | 0.79046458 | 0.0000872985 |
| MSHNet（CVPR24，val 第 100 epoch） | 0.29225813 | 0.45232160 | 0.80942706 | 0.0000672767 |
| UIU-Net（TIP23，val 第 20 epoch） | 0.21485370 | 0.35371124 | 0.68738995 | 0.0000296478 |
| SCTransNet（TGRS24，val 第 280 epoch） | 0.20290176 | 0.33735383 | 0.62210484 | 0.0000252791 |
| RPCANet（WACV24，val 第 40 epoch） | 0.30638360 | 0.46905611 | 0.80617635 | 0.0000775734 |
| HDNet（TGRS25，val 第 120 epoch） | 0.27259038 | 0.42840239 | 0.82134634 | 0.0000545049 |
| ILNet（TAES25，val 第 220 epoch） | 0.19172233 | 0.32175672 | 0.58946228 | 0.0000193025 |
| RPCANet++（TPAMI26，val 第 42 epoch） | 0.30094974 | 0.46266160 | 0.76919951 | 0.0000881637 |
| L2SKNet（TGRS25，val 第 130 epoch） | 0.30092370 | 0.46263082 | 0.77204388 | 0.0000862408 |
| BGM（TGRS25，val 第 50 epoch） | 0.30734902 | 0.47018663 | 0.78680753 | 0.0000963179 |
| DRPCA-Net（TGRS25，val 第 55 epoch） | 0.29985473 | 0.46136652 | 0.76852228 | 0.0001040385 |
| DURD（TGRS26，val 第 70 epoch） | 0.30060852 | 0.46225826 | 0.77583638 | 0.0001028447 |

三组预测目录为 `results/{DNANet,MSHNet,UIU-Net,SCTransNet,RPCANet,HDNet,ILNet,RPCANetpp,L2SKNet,BGM,DRPCANet,DURD}/Seal/fullsup_test_maps`，GT 计数与上表相同（7,383 个连通域、514,338,642 像素），因为用的是同一批 test GT 与同名原尺寸 PNG。完整计数见 `docs/<Method>_Seal评估产物/fullsup_small_metrics.json`。

这一组 Fa（1.9e-5～1.0e-4）比五个弱监督跨域方法低 2～3 个数量级，Pd 却高得多：全监督模型确实学到了目标响应，但预测连通域更"干净"；与 Ours（IoU 0.51931、F1 0.68361、Pd 0.89151、Fa 1.72e-05）相比，差距主要在 IoU/F1/Pd，Fa 上 Ours 只比最低的 ILNet（1.93e-05）低 11%。

## 待评估方法

WSSA、SCWSSOD、DENet、CRNet、RGBTScribble、WS-SAM、DHFR、SAM-COD、MIRV、RPPS、PSOD、P-COD 尚无主表原四项指标及对应的 Seal 测试预测，新增指标保留空白。后续完成各自协议下的测试后，使用同一预测同时补齐八项指标。
