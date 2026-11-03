# Seal 小目标指标评估

评估日期：2026-11-03。六组结果均使用量化主表中 S、Fw、E、MAE 对应的同一批 1,983 张预测；不重新选择权重，不重新推理。NESS-Net 复用已完成的同目录小目标评估结果。

统一入口为根目录 `evaluate_small_targets.py`，指标实现为 `metrics.py` 的 `AS1000Metrics`。定义见[小目标评价指标：定义与代码对照](小目标评价指标_定义与代码对照.md)。固定预测 >0.5、GT >128、8 连通、质心距离严格小于 3 像素的扫描顺序一对一匹配。IoU/F1 为累计像素计数；Pd 为检出目标数/GT 目标数；Fa 为未匹配预测连通域面积/总像素数。所有表值都是原始比例。

## 结果

| 方法 | IoU↑ | F1↑ | Pd↑ | Fa↓（原始比例） |
|---|---:|---:|---:|---:|
| FCT-SAM-WSCOD | 0.00067187 | 0.00134283 | 0.00257348 | 0.0346740446 |
| LRDNet | 0.00064195 | 0.00128307 | 0.00067723 | 0.0107905600 |
| MIR-Net | 0.00080272 | 0.00160416 | 0.00243803 | 0.0182267484 |
| SSFam | 0.00099438 | 0.00198679 | 0.28741704 | 0.0402274131 |
| NESS-Net | 0.25151885 | 0.40194177 | 0.56738453 | 0.0000779817 |
| Ours | 0.44835069 | 0.61911897 | 0.82730597 | 0.0000020998 |

每组 GT 共 7,383 个连通域、514,338,642 个像素。PNG 文件名及原尺寸须完全对应；不额外进行归一化、形态学处理或面积筛选。原四项指标及其固定函数保持不变。

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

Seal框监督适配；Canny/general；原Fw验证选择epoch5；T替代深度。

- 预测：`/share_datasets/Seal_Model_Reproduction/NESS-Net-main/experiments/Seal_box/canny_selected_original_eval/test_maps`。
- 原四项指标记录：`/share_datasets/Seal_Model_Reproduction/NESS-Net-main/experiments/Seal_box/canny_selected_original_eval/test_metrics.json`。
- 小目标完整结果：[NESS-Net.json](小目标指标评估_20261103/NESS-Net.json)。

### Ours

已有测试预测；RGB-T/Box。

- 预测：`/share_datasets/Seal_Model_Reproduction/Ours/test_maps`。
- 原四项指标记录：`/share_datasets/Seal_Model_Reproduction/Ours/unified_metrics.json`。
- 小目标完整结果：[Ours.json](小目标指标评估_20261103/Ours.json)。

## 复算

```bash
cd /share_datasets/Seal_Model_Reproduction
/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python evaluate_small_targets.py \
  --pred_root <上述预测目录> \
  --gt_dir /share_datasets/Seal_Model_Reproduction/Seal_data/test/GT \
  --json_output <结果JSON路径>
```

## 待评估方法

WSSA、SCWSSOD、DENet、CRNet、RGBTScribble、WS-SAM、DHFR、SAM-COD、MIRV、RPPS、PSOD、P-COD 尚无主表原四项指标及对应的 Seal 测试预测，新增指标保留空白。后续完成各自协议下的测试后，使用同一预测同时补齐八项指标。
