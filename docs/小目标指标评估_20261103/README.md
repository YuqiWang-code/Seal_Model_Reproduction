# 小目标指标评估（2026-11-03）

本目录保存六个方法在 Seal test 上的 `AS1000Metrics` 完整结果，供量化主表 IoU / F1 / Pd / Fa 四列与横向汇总引用。评估定义、文献对照与本目录的复算方式见：

- [小目标评价指标：定义与代码对照](../小目标评价指标_定义与代码对照.md)
- [Seal 小目标指标评估汇总](../Seal小目标指标评估汇总.md)

## 文件

| 文件 | 内容 |
|---|---|
| `FCT-SAM-WSCOD.json` | 公开检查点跨域测试，`results/FCT-SAM-WSCOD/Seal/test_maps` |
| `LRDNet.json` | S-COD 源域涂鸦训练（两阶段各 100 epoch）后跨域测试，`LRDNet-master/experiments/SCOD/test_maps` |
| `MIR-Net.json` | 公开完整权重跨域测试，`results/MIR-Net/Seal/published_test_maps` |
| `SSFam.json` | 公开 RGBT 完整权重跨域测试，`results/SSFam/Seal/RGBT_published` |
| `NESS-Net.json` | 公开 `NESS_refinement.pth` + NAMLab 边缘跨域测试，`results/NESS-Net/Seal/refinement_namlab_test_maps`；即主表 NESS-Net 行 |
| `NESS-Net_Seal框适配_epoch5.json` | Seal 框监督适配第 5 epoch 的历史结果，仅存档，不进入主表 |
| `Ours.json` | 已有测试预测，`Ours/test_maps` |
| `summary.json` | 上述六组（主表口径）的合并视图 |

每个 JSON 记录四项指标、像素/目标计数、完整协议参数、预测目录、GT 目录、方法名、实验协议及原四项指标来源。所有数值为 0～1 原始比例。

## 协议

预测恢复 GT 原尺寸并保存 8 位灰度 PNG 后评估：预测 `prob > 0.5`（即灰度 ≥128），GT `>128`，8 连通，质心距离严格小于 3 像素的栅格顺序一对一匹配。IoU / F1 为全数据集累计像素计数；Pd = 检出目标数 / GT 目标数；Fa = 未匹配预测连通域面积 / 总像素数。分母为零时返回 0。

六组均为 1,983 张原尺寸预测、7,383 个 GT 连通域、514,338,642 个像素（417×622 × 1,983）。

## 核对

- 每个 JSON 的四项指标都能由自身存储的计数精确复算：`python tmp/verify_small_target_jsons.py`。
- `docs/小目标评价指标.pdf` 草稿与现行实现的四处差异（阈值、Fa 分子、匹配边界、输出单位）及其数值影响已记录在[定义与代码对照](../小目标评价指标_定义与代码对照.md#与-docs小目标评价指标pdf-参考实现的差异)；表值只与同口径结果比较。
- 本目录与量化主表必须同步：主表 NESS-Net 行使用本目录 `NESS-Net.json` 的同一批预测，不得用框适配的存档结果填充。
