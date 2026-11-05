# Seal 数据集统计图

用于论文的数据集统计（对标 IRSTD 论文里的 Fig. 4 样式）。统计对象是服务器
`/share_datasets/Seal_Model_Reproduction/Seal_data/{train,val,test}/GT` 的全部
3,983 张 GT，沿用 Seal 原划分（1,500 / 500 / 1,983）。

## 定义

- 目标 = GT 二值化（`>127`）后的 **8 连通域**，一个连通域记一个目标/mask。
- 目标大小 = 该连通域包含的像素数。
- 每图目标个数 = 该图 GT 的连通域个数。所有图像至少有 1 个目标（无空 GT）。

## 统计结果

| 划分 | 图像 | 目标 | 目标大小 中位数 | 均值 | P99 | 最大 |
|---|---:|---:|---:|---:|---:|---:|
| train | 1,500 | 4,300 | 15 px | 16.94 px | 49.0 px | 76 px |
| val | 500 | 871 | 16 px | 20.07 px | 55.6 px | 72 px |
| test | 1,983 | 7,383 | 13 px | 14.32 px | 40.0 px | 114 px |
| **全部** | **3,983** | **12,554** | **14 px** | **15.62 px** | **47 px** | **114 px** |

每图目标个数（全部 3,983 张）：

| 目标数/图 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11–15 | 16–20 | >20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 图像数 | 1,782 | 836 | 410 | 244 | 194 | 112 | 81 | 61 | 47 | 26 | 101 | 31 | 58 |

95.2%（3,793 张）的图像最多 10 个目标，尾部单张最多 47 个。

## 版本与文件

两套配色都保留，图和 PDF 按版本放子目录，统计 JSON 在顶层共用：

| 目录 | 柱子颜色 |
|---|---|
| `v1/` | 深蓝 `#2E5E8C`（细边 `#1F3F5E`） |
| `v2/` | **橙色 `#E2903E`（细边 `#B26A20`）** |

每个版本目录里都是同样 4 张图，各有 PNG 与 PDF：

| 文件 | 内容 |
|---|---|
| `seal_dataset_statistics.png` / `.pdf` | **(a)(b) 双栏合并图，可直接进论文** |
| `seal_target_number_distribution.png` / `.pdf` | 单图 (a)：横轴每图目标个数 |
| `seal_target_size_distribution.png` / `.pdf` | 单图 (b)：横轴目标大小（像素） |
| `seal_target_scr_distribution.png` / `.pdf` | 单图：横轴目标 SCR、纵轴图片数量 |

PNG 为 300 dpi 位图；PDF 为矢量图，字体以 TrueType 子集嵌入（`pdf.fonttype=42`，不是部分期刊拒收的 Type 3），
可直接用于 LaTeX `\includegraphics`。

生成时用 `--accent` 选配色、`--outdir` 指定版本目录：

```bash
python tools/plot_seal_dataset_stats.py --accent blue   --outdir Ours/dataset_stats/v1
python tools/plot_seal_dataset_stats.py --accent orange --outdir Ours/dataset_stats/v2
```

## SCR 定义与结果

SCR = `|mean(target) − mean(background)| / std(background)`，在**原始 16 位热图**上算
（逐图 min-max 到 8 位是线性变换，不改变 SCR，但 16 位避免量化误差）。背景取目标包围框外扩一倍
（窗口边长 = 3 × 包围框边长）的局部区域、去掉目标像素；局部背景不足 20 像素或 std 为 0 时退回全图背景。
**每张图取该图所有目标 SCR 的最大值**，因此纵轴是「图片数量」；按目标统计的分布另存在 JSON 里。

| 划分 | 图片 | 图像 SCR 中位数 | 均值 | P95 | 最大 |
|---|---:|---:|---:|---:|---:|
| train | 1,500 | 4.08 | 4.85 | 11.50 | 21.34 |
| val | 500 | 4.84 | 5.79 | 15.61 | 24.58 |
| test | 1,983 | 5.16 | 5.66 | 11.60 | 20.03 |
| **全部** | **3,983** | **4.79** | **5.37** | **11.93** | **24.58** |

全部 3,983 张按 1 个 SCR 一格统计（图里 0–20，`>20` 的 7 张 / 0.18% 用角注）：峰值在 SCR 3–4（467 张），
SCR 1–6 区间集中 2,140 张（53.7%），SCR ≥ 10 的 448 张（11.2%）。

两栏都以**占比**为纵轴、右轴给**累计占比**（Fig. 4(b) 的做法）：

- (a) 纵轴 = 该目标个数的图像数 / 3,983（%）；右轴 = 累计图像占比。
- (b) 纵轴 = 该尺寸区间的目标数 / 12,554（%）；右轴 = 累计目标占比。
- 横轴分箱：(a) 只画 1–10（占 95.2%），其余 190 张（4.8%，最多 47 个）不画柱，
  在图中注明「>10 targets: 190 images (4.8 %), see text」，累计曲线止于 95.2%；
  (b) 5 像素一格、0–60；`>60` 的 30 个目标（0.24%，最大 114 px）同样不画柱、左上方注明，
  累计曲线止于 99.76%。两栏都不让长尾把坐标轴和分箱拉变形。
- 版式按**印刷尺寸**设计：合并图 7.8 × 2.75 in（双栏论文的整栏宽），两栏宽度比 **1 : 1.15**（右侧长尾更舒展），
  字号 7.5–8.5 pt，PDF 里的 pt 就是最终 pt；单图 (a) 3.9 in、(b) 4.2 in 宽。合并图用顶部居中的共享图例。
- 配色与细节：**单色系深蓝**——柱 `#2E5E8C` 填充 + `#1F3F5E` 细边（0.6 pt）；累计曲线浅蓝 `#6FA8D0`、
  1.35 pt，下方垫一条 2.2 pt 白色衬线，使其压过深色柱时仍可读；右轴文字用较深的蓝 `#4A7FA8`；
  仅左/下轴 + 淡横向网格（`#EAEAEA`，0.5 pt）；轴标签与刻度深灰 `#333/#4A4A4A`；
  字体 **Microsoft YaHei**，`bbox_inches='tight'`。

图注可用：

> Fig. X. Statistics of the Seal dataset. (a) Target number per image and (b) target size
> distribution. Bars: percentage of images / targets; blue lines (right axis): cumulative percentage.

## 复算

统计在服务器上只读 GT 计算（本地只有 test 的 GT），绘图在本地：

```bash
# 服务器（任意 python3 + opencv）
ssh RSML-3 "python /share_datasets/Seal_Model_Reproduction/tools/compute_seal_dataset_stats.py" \
  > Ours/dataset_stats/seal_dataset_stats.json
ssh RSML-3 "python /share_datasets/Seal_Model_Reproduction/tools/compute_seal_scr.py" \
  > Ours/dataset_stats/seal_scr_stats.json

# 本地
python tools/plot_seal_dataset_stats.py                 # 全部 3,983 张，默认 v1 输出到 ./Ours/dataset_stats
python tools/plot_seal_dataset_stats.py --split test        # 只看 test
```

`seal_dataset_stats.json` 保存每个划分与全部的逐值直方图（`per_image`、`area_hist`）
及面积汇总，重绘不依赖原始 GT。
