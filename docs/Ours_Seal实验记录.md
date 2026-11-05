# Ours：Seal 测试预测与统一评估

## 数据与预测

- 预测来源：**2026-11-05 起的 1105 批次**，`Ours/1105Ours_result/Ours.zip` 解压至 `Ours/test_maps/`，共 1,983 张 8 位灰度 PNG（文件名与 `Seal_data/test/GT` 一一对应，尺寸 417×622）。此前使用的旧预测（原 `Ours/test_maps.zip`）改名存档为 `Ours/test_maps_20261105_prev`，仅作历史对照。
- GT 来源：根目录 `Seal.zip` 中的 `Seal/test/GT/`，提取至 `Ours/Seal_test_GT/`。
- 预测与 GT 的 1,983 个文件名完全对应，所有图像尺寸均为 417×622，无尺寸不匹配，无空 GT。
- 表格将 Ours 标为 RGB-T、Box。本次使用已有预测进行评估，不重新训练或推理。
- 评估日期：旧预测 2026-11-03，1105 批次 2026-11-05。四项指标直接调用 `Ours/evaluate.py` 的四个固定原函数计算，每个函数均评估 1,983 对预测与 GT。E 为原函数对连续灰度预测计算的结果。

## 最终代码与复现（`Ours/CODE/`）

最终代码在 `Ours/CODE/`，与 `Ours/1105Ours_result/Ours.zip` 一一对应；本表第 20 行、定性图（v12）、ROC/P-R（v10/v11）与目标级 Pd–Fa（v12）都用这批预测。代码内的指标实现与项目权威版本逐字节相同：`evaluate.py` = `Ours/evaluate.py`，`metrics.py` = 根目录 `metrics.py`（四个原函数与 `AS1000Metrics` 均未改）。

| 文件 | 作用 |
|---|---|
| `CM-LoRA.py` | 冻结的 SAM3 + 两套视觉模态 LoRA（`outputs/{RGB+LoRA,T+LoRA}/best/lora_weights.pt`），共享 LoRA3 挂到同一批 qkv/proj 线性层（rank 8、α 16），不注册进 SAM3 |
| `BISF.py` | `M1` 双向交互空间融合：两路 backbone 最后一层做双向空间交叉注意力 + 深度卷积 + 空间自注意力，`fused = (R+T)/2 + Conv1x1(cat(R′,T′))`，1×1 权重与偏置零初始化 |
| `train.py` | 训练入口：只训练 `M1`（1,191,168 参数）与共享 LoRA3（1,572,864），SAM3 与模态 LoRA 冻结；损失 = 结构损失 + Sigmoid Focal（α=0.25、γ=2）+ Dice，等权；AdamW lr 5e-5、weight decay 0.01；默认 30 epoch、每卡 batch 1、1008×1008 |
| `test.py` | 推理入口：加载 `best.pth`（`m1`+`lora3`）+ `sam3.pt` + 两套模态 LoRA；用数据集里的**真实框**作 Decoder 几何提示；1008² 前向 → sigmoid → 双线性恢复 GT 尺寸 → 逐图 min-max → 8 位 PNG |
| `data.py`、`utils.py` | Seal 数据集（16 位 `T` 逐图 min-max → 8 位）、增强、框/掩码处理与概率后处理 |
| `evaluate.py`、`evaluate3.py`、`metrics.py` | 四个原函数、八项合并评估、`AS1000Metrics` |
| `efficiency.py` | 参数 / FLOPs / FPS 统计 |

复现命令（服务器，在 `Ours/CODE/` 下）：

```bash
# 训练（3 卡，每卡 batch 1）
python train.py --epoch 30 --lr 5e-5 --img_size 1008 \
  --train_data_dir /share_datasets/Seal_Model_Reproduction/Seal_data/train \
  --val_data_dir   /share_datasets/Seal_Model_Reproduction/Seal_data/val \
  --save_path train_logs
# 测试：输出与 GT 同名的 8 位 PNG
python test.py --checkpoint train_logs/best.pth --save_dir test_maps \
  --test_data_dir /share_datasets/Seal_Model_Reproduction/Seal_data/test
# 八项指标（权威口径用项目根目录入口，显式传 --gt_dir）
python /share_datasets/Seal_Model_Reproduction/evaluate.py --pred_root test_maps \
  --gt_dir /share_datasets/Seal_Model_Reproduction/Seal_data/test/GT
python /share_datasets/Seal_Model_Reproduction/evaluate_small_targets.py --pred_root test_maps \
  --gt_dir /share_datasets/Seal_Model_Reproduction/Seal_data/test/GT --json_output small.json
```

消融开关：`--no-lora3`（只训 M1）、`--no-lora`（RGB/T/LoRA3 都不用，共用冻结 SAM3）、`--no-m1`（融合退化为 `(RGB+T)/2`）。代码目录 `ablation/LoRA/{RGB,RGB+LoRA,T,T+LoRA,RGB-T,RGB-T+LoRA,3LoRA}/` 存各消融的 1,983 张预测，对应表内 `Ablation` 工作表。

权重与源码：`outputs/{RGB+LoRA,T+LoRA}/best/lora_weights.pt`（各 12.11 MB）、`3LoRA/best.pth`（6.03 MB）、`train_logs2/best.pth`（10.59 MB，`m1`+`lora3`）均已入库；SAM3 主干 `sam3.pt`（3.21 GiB）只在服务器 `Ours/weights/`；`CODE/sam3/`、`CODE/SAM3_LoRA/` 是随代码带的 SAM3 源码树（本地约 106 MB），按 `.gitignore` 不入库，服务器保留完整副本。

**批次核对（2026-11-05）**：`Ours/1105Ours_result/Ours.zip`、`Ours/CODE/test_maps/`、`Ours/test_maps/` 三处的 1,983 张预测逐字节相同（sha256 前缀 `22b95e54`）。用根目录入口复算得到 S 0.4995640741534494、Fw 0.6196639054873058、E 0.6969485079429705、MAE 0.00020247850625310093、IoU 0.5193093402494049、F1 0.6836123842484333、Pd 0.8915075172694027，与第 20 行一致（S 仅在第 11 位差 5×10⁻¹¹）。旧预测（S 0.49915、Fw 0.67338、E 0.66731、MAE 0.00015、IoU 0.44835）已移到 `Ours/test_maps_20261105_prev/`。

## 统一评估结果

现行口径为 2026-11-05 的 **1105 批次**，定性图、ROC/P-R、目标级 Pd–Fa 曲线与量化表第 20 行均用这批预测：

| 结果来源 | S↑ | Fw↑ | E↑ | MAE↓ |
|---|---:|---:|---:|---:|
| 指定四个原函数，1105 批次（现行） | 0.49956 | 0.61966 | 0.69695 | 0.00020248 |
| 旧预测（`test_maps_20261105_prev`，历史） | 0.49915 | 0.67338 | 0.66731 | 0.00015 |
| 腾讯表格原有记录 | 0.49781 | 0.51296 | 0.64987 | 0.00055 |
| 其他评估口径，存档 | 0.80498 | 0.67439 | 0.94830 | 0.00015 |

量化 SOTA 表采用指定四个原函数的完整结果，不根据数值大小混用不同口径。1105 批次的完整精度见 [`ours1105_metrics.json`](Ours_Seal评估产物/ours1105_metrics.json)（服务器 `results/ours1105_metrics.json`），旧预测见 [`Ours/unified_metrics.json`](../Ours/unified_metrics.json)；其他口径结果保存在 [`Ours/other_protocol_metrics.json`](../Ours/other_protocol_metrics.json)。两批预测的差异：1105 批次的 Fw 与 MAE 略差（误检更多、背景响应更弥散），但 IoU/F1/Pd 明显更好，Fa 由 2.10e-06 升到 1.72e-05。

根目录 `evaluate.py` 的四个指标函数与 `Ours/evaluate.py` 原文一致，入口额外接收 GT 路径。两种评估口径在 S 的区域计算、Fw 的高斯平滑及 E 的计算方式上不同，MAE 的实现相同。

在项目根目录复算：

```bash
python evaluate.py \
  --pred_root Ours/test_maps \
  --gt_dir Ours/Seal_test_GT
```

## 预测分析

面积占比为逐图计算后平均；precision 和 recall 同时列出逐图平均与汇总全部像素后的结果。阈值统计使用灰度值大于 128 的像素，不用于替代连续灰度预测的 Fw 或 MAE。

| 项目 | 1105 批次（现行） | 旧预测（历史） |
|---|---:|---:|
| GT 平均前景面积占比 | 0.020553% | 0.020553% |
| 预测平均前景面积占比，阈值 128 | 0.029333% | 0.018883% |
| precision，逐图平均 | 61.88% | 65.53% |
| recall，逐图平均 | 84.64% | 60.62% |
| precision，汇总全部像素 | 58.13% | 64.64% |
| recall，汇总全部像素 | 82.96% | 59.39% |
| 阈值以上无预测前景的图像 | 0 / 1,983 | 19 / 1,983 |
| 预测中灰度值为 0 的像素占比 | 99.932309% | 99.975767% |
| 预测中灰度值为 1～254 的像素占比 | 0.061596% | 0.019607% |

1105 批次完整统计见 [`Ours_Seal评估产物/ours1105_prediction_analysis.json`](Ours_Seal评估产物/ours1105_prediction_analysis.json)（服务器 `results/ours1105_prediction_analysis.json`），旧预测见 [`Ours/prediction_analysis.json`](../Ours/prediction_analysis.json)。1105 批次的预测前景面积比 GT 大 43%（0.0293% vs 0.0206%），所有图像都有前景响应、召回明显更高，代价是背景响应更弥散（非零非满像素占比升到 0.062%），因此 Fw 与 MAE 相对旧预测变差。对比方法在 Seal 上的大量背景误检会同时拉低 Fw、抬高 MAE；这解释了统一评估后仍存在的量级差异。

Seal GT 的平均前景面积占比为 0.00020553，因此全黑预测的平均 MAE 也为 0.00020553。1105 批次 Ours 的实际 MAE 为 0.00020248，仅比该全背景基线低 1.5%（旧预测 0.00015090 低 27%），并具有 0.61966 的 Fw。

## RGB 与热图可视化

服务器归档：`/share_datasets/Seal_Model_Reproduction/Ours/vis.zip`，压缩包为 1,050,737,442 字节。大文件仅保留服务器副本。

| 目录 | 数量 | 内容 |
|---|---:|---|
| `vis/RGB/` | 3,983 | RGB 场景可视化，样例带框与类别标记 |
| `vis/T/` | 3,983 | 热图可视化，样例带框与类别标记 |
| `vis/T0/` | 3,983 | 热图可视化，样例不带框与类别标记 |

三组均为 JPG，编号覆盖 1～3,983，包含完整测试编号 2,001～3,983。它们可用于核对两路图像的目标位置、检查微小目标的可见性，并结合 GT 和预测图展示误检、漏检。样本 2001 的图像尺寸为 417×622，GT 包围框为 4×8 像素。

本地保留该样本的三张小图：[RGB](../Ours/vis_samples/2001_RGB.jpg)、[T](../Ours/vis_samples/2001_T.jpg)、[T0](../Ours/vis_samples/2001_T0.jpg)。

这些导出的 JPG 用于展示，不替代原始 16 位热图、框坐标或二值 GT。压缩包未提供 T/T0 的转换代码，不能仅凭显示效果确定其数值变换方式。

## 已知问题

- **最终检查点用 Seal test 选点（须重跑）**：`Ours/CODE/train_logs2/train.log` 记录本次训练（2026-11-05 11:28 起，30 epoch）的 `val_data_dir=/data/RGB-T-dataset/Seal/test`、`samples: 1983`、`bestIoU 0.51925 bestEpoch 3`。第 20 行的 test IoU 0.51931 与该"验证"值几乎相同，说明第 20 行的检查点按 Seal **test** 选择。这违反「不根据 Seal 测试集指标选择权重」。投稿前须把 `--val_data_dir` 指向 `Seal_data/val`（500 张）、按 val 像素 IoU 重选检查点，再用 `test.py` 重新推理，并同步第 20 行的八项指标、定性图（现 v12）、ROC/P-R（现 v10/v11）与目标级 Pd–Fa（现 v12）。当前数值如实保留并在此标注，不做任何掩盖。
- `Ours/CODE/evaluate3.py` 的 GT 路径硬编码为作者本地的 `/data/RGB-T-dataset/Seal/test/GT`；在本仓库/服务器上须改参数或直接用根目录 `evaluate.py` + `evaluate_small_targets.py`（显式 `--gt_dir`）。权威口径始终是根目录入口，`evaluate3.py` 只作为作者侧八项合并结果的对照。
- `test.py` 的默认 `--checkpoint` 是 `train_logs/best.pth`，本次最终权重实际在 `train_logs2/best.pth`，运行时需显式指定；默认 `--test_data_dir`、`--sam_checkpoint` 也是作者本地路径，服务器上应换成 `/share_datasets/Seal_Model_Reproduction/...`。
- 预测包本体未附训练配置快照；本文依据的是 `Ours/CODE/train_logs2/train.log` 的实参记录（epoch 30、lr 5e-5、wd 0.01、batch 1/卡、1008²、3 卡）与 `best.pth` 的键（`m1`+`lora3`）。
- 使用指定原函数后，结果仍未完全复现腾讯表格旧值。两种口径的 MAE 实现相同，旧预测均得到 0.00015090（1105 批次为 0.00020248）；原表 MAE 0.00055 的差异不能由此次公式切换解释。原表未附预测、GT 路径及评估日志，具体输入差异尚不能确定。

## 小目标指标（同批预测）

2026-11-05，在 1105 批次的 1,983 张测试预测上用统一 `AS1000Metrics` 计算（阈值 >0.5、8 连通、质心距离 <3 px、全数据集累计）。预测目录：`/share_datasets/Seal_Model_Reproduction/Ours/test_maps`。

| 批次 | IoU↑ | F1↑ | Pd↑ | Fa↓（原始比例） |
|---|---:|---:|---:|---:|
| 1105（现行，进表） | 0.51931 | 0.68361 | 0.89151 | 0.0000171735 |
| 旧预测（历史） | 0.44835 | 0.61912 | 0.82731 | 0.0000020998 |

1105 批次计数：tp 87,702、fp 63,170、fn 18,010、n_gt 7,383、n_pred 6,853、n_detected 6,582、false_alarm_pixels 8,833、total_pixels 514,338,642。完整精度、计数及协议见 [`ours1105_small.json`](Ours_Seal评估产物/ours1105_small.json)（服务器 `results/ours1105_small.json`）；旧预测见 [Ours.json](小目标指标评估_20261103/Ours.json)；横向结果见[Seal 小目标指标评估](Seal小目标指标评估汇总.md)。

## 特征响应图

### v1（第一版，`Ours/Feature_Map/v1/`，历史）

用最终权重（`Ours/CODE/train_logs2/best.pth` + 两套模态 LoRA + 共享 LoRA3 + `M1`/BSRF）为 10 个样例出 9 列响应图：

| 列 | 内容 |
|---|---|
| 1–2 | RGB 原图、T 原图（16 位逐图 min-max → 8 位） |
| 3–4 | **w/o CM-LoRA**：冻结 SAM3（`build_frozen_base_model`，无模态 LoRA、无 LoRA3、无 BSRF）的 backbone 最后一层响应 |
| 5–6 | **w/ CM-LoRA**：两套模态 LoRA + 共享 LoRA3，取 BSRF **之前**的 backbone 最后一层响应 |
| 7 | **BSRF 融合响应**：`M1(rgb_out, t_out)` 的 `backbone_fpn[-1]` |
| 8 | 最终软预测（0–1 概率伪彩，与 `test.py` 同一路径：top-k logits → sigmoid → 恢复 GT 尺寸） |
| 9 | GT mask |

- 响应图取 `backbone_fpn[-1]`（256×72×72）的**通道 L2 范数**，按 2–98 百分位归一化后套 TURBO 伪彩；不用通道均值（bd16 特征均值会被均匀分量主导、目标被抹平）。
- 样例按**逐图 F1**降序选前 10（`Ours/test_maps` 与 test GT 在 ≥128 阈值下逐图统计，同时记录 IoU）：2105（1.00000/1.00000）、3343（0.96774/0.93750）、2146（0.96552/0.93333）、2059（0.95146/0.90741）、3000（0.94737/0.90000）、3173（0.94737/0.90000）、2773（0.93506/0.87805）、3174（0.93333/0.87500）、2754（0.92857/0.86667）、3517（0.92308/0.85714）；完整排名见服务器 `ranking.json`。
- 每张图三套输出：`full/`（原尺寸 417×622）、`roi/`（以 GT 外接框为中心、边长 `max(64, max(bbox)+2×24)` 并夹在图像内，×6 放大）与 `manifest.json`；Visio 用 `roi/`。
- 产物：服务器 `Ours/Feature_Map/v1/`：`panels/<order>_<stem>/{full,roi}/01_rgb … 09_gt.png`（180 张）、`manifest.json`、`ranking.json`、`Ours_FeatureMap_v1.vsdx`（23.05 MB，一页 10 行 × 9 列，行首标 stem + F1/IoU，表头按列分组配色）；本地 `Ours/Feature_Map/v1/` **只放该 vsdx**，`Ours/Feature_Map/` 与 `tmp/feature_v1/` 按 `.gitignore` 不入库。
- 复现：服务器 `CUDA_VISIBLE_DEVICES=1 Ours/venv_ours/bin/python tools/feature_maps.py --out Ours/Feature_Map/v1 --topk 10 --rank-by f1`（脚本 `tools/feature_maps.py`，日志 `logs/feature_maps_v1.log`）；Visio 由 `tools/make_visio_feature_maps.ps1`（Visio COM）从 `panels/` 重新生成。
- 读数提示：前 8 行 ROI 均为 64×64（×6 后 384×384），第 10 行 3517 的 GT 较大、ROI 为 182×182，特征格相对更细；响应峰位置与第 8 列预测、第 9 列 GT 一致。

### v2（已被 v3 取代，`Ours/Feature_Map/v2/`）

v2 与 v1 的差别在于**把 03–06 换成各模态自己的解码输出**（而不是 backbone 特征），只保留 07 作为特征图，这样九列讲的是同一条链路：单模态原始输出 → 单模态适配输出 → 跨模态融合特征 → 最终输出 → GT。

| 列 | 内容 | 应呈现的特质 |
|---|---|---|
| 1–2 | RGB / T 原图（ROI ×6，1 px 青框标 GT） | 目标 3–5 px、RGB 常与背景同色，T 上更亮；框用来快速定位 |
| 3–4 | `decode(冻结 SAM3, 单模态特征, box)` 的 sigmoid 概率 | 原始 SAM 只靠单模态：位置可能对，但形状散、常有远处误检（本批 median IoU RGB 0.504 / T 0.607，3343、3173 的 RGB 分支直接为 0） |
| 5–6 | 同上，但换成 LoRA+LoRA3 的单模态模型（BSRF 之前） | CM-LoRA 应把响应收紧到目标附近：median IoU RGB 0.884 / T 0.762；最明显的是 RGB 分支（0.00→0.94、0.00→0.90） |
| 7 | `M1` 融合特征 `backbone_fpn[-1]`（72×72）的通道 L2 | 两路信息的互补结果：单个热点覆盖目标，背景被压低（峰值 10/10 落在 GT 框内，距质心中位 2.28 px，inside/outside 中位 1.64×） |
| 8 | 最终融合预测（与 `test.py` 同一路径） | 应等于或优于任一单模态分支：median IoU 0.889，范围 0.818–1.000 |
| 9 | GT mask | 参照 |

处理原则（为什么这么做）：

1. **概率类统一 0–1 固定色标**（03–06、08）：v1 逐图 min-max 会把弱预测也拉成满色，跨列看不出"谁更可信"；固定标度后颜色直接对应概率。
2. **03–06 用单模态解码输出而不是 backbone 特征**：主干末层是 72×72（stride≈14 px/格），目标不足 1 格，特征图无法代表"该模态判成什么样"；单模态解码才有 IoU 可量、可比较。07 保留特征，因为这一层正是 BSRF 融合的那一层。
3. **07 用通道 L2 + 逐图 2–98% 归一化**：融合层分辨率同上，画出来是**感受野热点**而非 mask；逐图归一化保证形状可读，绝对强度靠 caption 的 peak ratio。改用全局标度会把各图压成一片橙红（已试过，不可读）。
4. **聚合方式的实测比较**（`tools/feature_maps_analyze.py` → `v1/aggregation_stats.json`）：通道均值被 DC 分量淹没（peak-in-GT 0–1/10）、PCA1 抓到的是背景方差方向（0/10）、通道 max 略好但噪声大、**通道 L2 的 contrast 与局部化最平衡**，故用 L2；"按 GT 选通道的 top-k"peak 命中 8/9 但用了 GT，只能当可视化、不能当证据，未采用。
5. **ROI 与标注**：ROI 仍 6×（边长 `max(64, bbox+2×24)`，夹在图像内）；每行左侧写 final F1/IoU，每列下方写该分支 IoU，07 下方写 peak 距离与 inside/outside 比。

产物：服务器 `Ours/Feature_Map/v2/`（`panels/<order>_<stem>/{full,roi}/01–09.png` 共 180 张、`legend_prob.png`、`legend_feature.png`、`manifest.json`、`Ours_FeatureMap_v2.vsdx` 9.2 MB）；本地 `Ours/Feature_Map/v2/` 只放该 vsdx。重建：`tools/feature_maps_v2.py` + `tools/make_visio_feature_maps_v2.ps1`。

### v3（已被 v4 取代，`Ours/Feature_Map/v3/`）

v3 只改**呈现**：模型、权重、预测都没动（同一 `Ours/CODE/train_logs2/best.pth`、同一 1105 预测），概率面板的 IoU 与 v2 逐位相同。

要解决的问题：v2 的第 7 列比第 8 列更大更糊，读起来像"融合之后反而退化"。根因是分辨率而不是模型——融合层 `backbone_fpn[-1]` 是 72×72，一格约 9×6 原生像素，而目标只有 3–5 px，所以按 0–1 全量上色必然得到一块比预测还大的色斑。实测这条响应是一个平滑峰：50% 水平半径中位 15.2 px，但 90% 水平只有 3.6 px——"清晰"不在低阈值上，在峰顶。

| 等值线水平 | 0.5 | 0.7 | 0.8 | 0.9 |
|---|---:|---:|---:|---:|
| 半径中位（px，10 张） | 15.2 | 8.2 | 6.2 | 3.6 |

做法：

1. **07 只画峰顶**：`rel = (v - median) / (peak - median)`，显示窗口 rel 0.85–1（γ 1.5），白线为 rel 0.9 等值线（半径约 3.6 px），白色十字为 argmax。第 7 列因此变成"目标框里的一个紧凑峰"，与第 8 列同量级。
2. **03–06、08 保持固定 0–1 概率色标**，并叠加 0.5 白等值线，也就是算 IoU 的那块区域；读图时"白线圈出的面积"从散 / 无 → 一个目标 → 精确边界。
3. **GT 框统一成青色虚线**（在放大后的面板上按 1 px 画，两张原图仍是青色实线）：青色 = 参考位置，白色 = 响应，两者不再混。
4. **ROI 收到 `max(44, bbox + 28)`**，放大倍数按面板统一到 396 px（本批全部 44 px / 9×），10 行可直接比较。
5. **样例改为"逐图 F1 前 10 且 GT 框 ≤ 32 px"**：v2 的第 10 张 3517 GT 框 128 px，ROI 与倍数都和其它行不同，峰值距 GT 质心 108 px，放进"小目标"图里会误导；改取排名第 11 的 3974。10 张为 2105、3343、2146、2059、3000、3173、2773、3174、2754、3974。
6. **底部加中位数行**：raw RGB/T 0.63 / 0.61，CM-LoRA RGB/T 0.88 / 0.78，final 0.89；BSRF 峰距 1.7 px、inside/outside 1.67×。

读数（10 张，1105 预测）：融合特征峰值 **10/10 落在 GT 框内**，距质心中位 1.7 px，目标内外强度比中位 1.67×；最终预测 median IoU 0.889、最低 0.818；CM-LoRA 把 RGB 分支 0.63 → 0.88、T 分支 0.61 → 0.78，与 v2 集合（0.50 → 0.88、0.61 → 0.76）同向。

局限（须写进图注）：07 是**峰**不是掩膜，低阈值上的宽度受 72×72 分辨率限制（50% 水平半径中位 15 px），不能用来论证边界精度；10 张是按 F1 选出的头部样本，不是随机抽样。

产物：服务器 `Ours/Feature_Map/v3/`（`panels/<order>_<stem>/{full,roi}/01–09.png` 共 180 张、`raw/<stem>.npz` 缓存、`manifest.json`、两条色标、`Ours_FeatureMap_v3.vsdx`）；本地 `Ours/Feature_Map/v3/` 只放 vsdx。重建：`tools/feature_maps_v3.py`（GPU 约 40 s；`raw/*.npz` 存了响应与四路概率，只改显示时不必再跑模型）+ `tools/make_visio_feature_maps_v3.ps1`（Visio COM，单页 11.15×15.41 in、191 个形状）。

### v4（已被 v5 取代，`Ours/Feature_Map/v4/`）

**为什么重做**：v3 的 10 行取的是"逐图 F1 前 10"，即最容易的一批；它们的 median IoU 0.63 / 0.61 / 0.88 / 0.78 / 0.89 不能当作方法水平。v4 先扫全测试集（`--scan-out`，1,983 张，GPU 约 25 min → `Ours/Feature_Map/v4/scan.csv`），拿到每张图每分支的 IoU 与峰值位置，再按**声明过的规则**取样、并按真实误差上色。

全测试集 GT 框 ≤ 32 px 的子集（n = 1,144）中位 IoU：冻结 SAM3 RGB **0.421** / T **0.429**，CM-LoRA RGB **0.556** / T **0.478**，最终 **0.565**；融合特征峰值落在 GT 框内 **84.1 %**，偏移中位 3.05 px。（全部 1,983 张：0.375 / 0.409 / 0.544 / 0.458 / 0.558。）v3 那组 0.63 → 0.88 → 0.89 是最好的十张，v4 把两个口径并列写在页脚，避免"拿最好样本当平均"。

**取样规则**（写进 manifest 的 `selection`）：GT 框 ≤ 32 px 且最终 IoU ≥ 0.8 的样本（n = 30）里，按冻结 SAM3 RGB 的 IoU 升序等距取 10 张——10 行因此覆盖"冻结模型完全失败（0.00）"到"冻结模型已经够好（0.88）"的整条轴，并保留反例（2498：LoRA 0.80 → 0.50）。行序即该轴升序。

**呈现改动**：

1. **不注入任何噪声**。实测冻结 SAM3 的失败形态不是散点噪声，而是**覆盖过大 / 位置偏移**：7–15 px 的目标上 FP 面积 18–163 px，而 CM-LoRA 降到 0–3 px。v4 直接画这件事——品红 = 预测落在 GT 外（FP），绿 = GT 未被覆盖（FN），青 = GT 边界，白 = 该列在 p = 0.5 的判定边。四种颜色都只由"预测 vs GT"决定，没有人工加工。
2. 每列标注改成 `IoU x.xx  FP nn`，把"多出来的响应"量化。
3. 07 沿用 v3 的峰顶窗口（rel 0.85–1，γ 1.5）+ 0.9 等值线 + 峰十字。
4. 中位数行明确标注为"这 10 张"；全测试集数字单独放在页脚粗体行。
5. 保留 `--auto-roi`（44–88 px，本批未触发）与 `--plain-heat`（退回无着色版本）两个开关。

**10 行的读数**：raw RGB 中位 0.66 → CM-LoRA RGB 0.87 → final 0.84（该子集 final 中位略低于 RGB 分支，见下）；最坏 FP 面积 163 px → 1 px；融合峰 10/10 在 GT 内，偏移中位 1.8 px。

**必须如实写进图注的三点**：

- T 分支的 LoRA 增益不稳定（示例里 4 张升、3 张降），不能把两个模态写成同等受益；
- 融合（BSRF）相对 RGB 单分支只有约 +1 个点（0.556 → 0.565），它的作用是"把峰值定到目标上"，不是大幅提分；
- 10 行是"已解决样本"的分层抽样，不是随机样本，全测试集分布见页脚。

产物：服务器 `Ours/Feature_Map/v4/`（`panels/` 180 张 396×396、`raw/<stem>.npz`、`scan.csv`、`scan_summary.json`、`manifest.json`、两条色标、`Ours_FeatureMap_v4.vsdx`）；本地 `Ours/Feature_Map/v4/` 只放 vsdx。重建：`tools/feature_maps_v4.py`（`--scan-out` 全量扫描 / `--stems` 指定样例，GPU 每 10 张约 5 s）+ `tools/make_visio_feature_maps_v4.ps1`（192 个形状，11.15×15.77 in）。

### v5（特征取对应热图版，已被 v6 取代；`Ours/Feature_Map/v5/`）

**版式**：只留列名 + 每行（原图 / 热力图）+ 底部两段色标——没有标题、箭头带、阶段小字、逐格注记、中位数行、页脚长文；左列只写样例编号。九列仍是 RGB、T、SAM3 (RGB)、SAM3 (T)、CM-LoRA (RGB)、CM-LoRA (T)、BSRF 特征、最终预测、GT。03–06 / 08 是各分支的解码概率（固定 0–1，显示 gamma 0.75，全列同一条曲线），07 是融合层通道 L2（峰顶窗口 85–100 %、γ 1.5）。`--annotate` 可恢复 v4 的等值线/误差着色，`--prob-gamma` 调色调。

**取样**：从"GT 框 ≤ 32 px 且最终 IoU ≥ 0.8"的 30 张里，按**冻结 SAM3 RGB 的预测面积 ÷ CM-LoRA RGB 的预测面积**降序取 10 张，即冻结模型过覆盖最严重的十例（面积比 1.4–284×，最小 2802、最大 2845：raw 1990 px vs LoRA 7 px，而 GT 只有 8 px）。这样每行都能在热图上直接读出"大面积弥散 → 紧致单块 → 单峰 → 目标形状"。规则写进 manifest 的 `selection`。

| 样例 | 2845 | 3343 | 3372 | 3315 | 3173 | 2105 | 3842 | 2754 | 3000 | 2802 |
|---|---|---|---|---|---|---|---|---|---|---|
| raw RGB 预测面积 (px) | 1990 | 163 | 108 | 48 | 38 | 25 | 18 | 28 | 18 | 32 |
| CM-LoRA RGB (px) | 7 | 16 | 19 | 8 | 10 | 7 | 8 | 14 | 11 | 23 |
| GT (px) | 8 | 15 | 17 | 10 | 9 | 7 | 12 | 13 | 9 | 21 |

**必须写明**：这十行是**冻结模型失败最严重**的十例，不是随机样本；全测试集分布仍是 0.421 → 0.556 → 0.565（见 v4 小节），正文引用时要说清是 worst-case 展示。

产物：服务器 `Ours/Feature_Map/v5/`（`panels/` 180 张 396×396、`manifest.json`、两条色标、`Ours_FeatureMap_v5.vsdx`；本地 `Ours/Feature_Map/v5/` 只放 vsdx）。重建：`tools/feature_maps_v5.py --stems 2845,3343,3372,3315,3173,2105,3842,2754,3000,2802` + `tools/make_visio_feature_maps_v5.ps1`（113 个形状，11.25×12.52 in）。

### v6（现行，`Ours/Feature_Map/v6/`）

**内容回到 v1**：03–07 各列是**主干特征**的热力图（`backbone_fpn[-1]` 的通道 L2，逐图 2–98 % 拉伸后上色），这正是 v1 好看的原因——特征图有结构，而解码概率本身就接近二值。列还是九列：`RGB / T / w/o CM-LoRA (RGB) / w/o CM-LoRA (T) / w/ CM-LoRA (RGB) / w/ CM-LoRA (T) / BSRF fused / Prediction / GT`；08 是最终预测（固定 0–1），09 是 GT，全图**没有任何标注或注记**。

相对 v1 只做了"稍微处理"：

1. **几何统一**：ROI = `max(64, GT bbox + 2×24)`、面板 384 px（本批全部 64 px / 6×）。v1 因为收了一个 182 px 的大目标（3517），那一行倍率与其它行不同。
2. **重新选样**（`--stems`，写进 manifest 的 `selection`）：在"GT bbox ≤ 32 px 且最终 IoU ≥ 0.8 且融合特征峰值距 GT 质心 ≤ 6 px"的 28 张里，按**融合特征的靶内外强度比**（1.56–1.90）等距取 10 张：3974、3372、2180、2732、2498、2226、2931、2773、2146、2105。行序即该比值降序，所以上面几行的热点更集中。
3. **版式干净**：只有列名 + 每行（左侧 `stem / F1 / IoU`）+ 底部一段两截色标（特征逐图归一、预测固定 0–1）。
4. 逐图响应图与预测缓存到 `raw/<stem>.npz`，以后只改显示不必再跑 GPU。

**显示曲线**：逐图 2–98 % 拉伸之后再做一次**顶端 15 % 窗口**（`--resp-window 0.85`：低于 85 % 的值压到暗蓝端，只有最高的一截留红），于是每格是"深蓝底 + 一块紧凑红核"而不是整片暖色。窗口是单调映射，不改变峰值位置，也**对五列特征用同一条**——只给某一列换曲线会让它和左右列不可比，所以没有这么做。`--resp-gamma`（默认 1.0）可在窗口后再加一条曲线。

**读图限制**：特征列是**逐图 2–98 % 归一化 + 顶端 15 % 窗口**，颜色只表示该图内的相对强弱，跨行、跨列都不能比较绝对强度；预测列是固定 0–1，两者不要混读。十行是按"融合特征集中度"分层抽的已解决样本，不是随机样本。

产物：服务器 `Ours/Feature_Map/v6/`（`panels/` 180 张 384×384、`raw/`、`manifest.json`、两条色标、`Ours_FeatureMap_v6.vsdx`）；本地 `Ours/Feature_Map/v6/` 只放 vsdx。重建：`tools/feature_maps_v6.py --stems 3974,3372,2180,2732,2498,2226,2931,2773,2146,2105`（默认 ROI 64 / 6×，`--roi-min/--roi-margin/--panel-px` 可调）+ `tools/make_visio_feature_maps_v6.ps1`（113 个形状，11.31×12.46 in）。
