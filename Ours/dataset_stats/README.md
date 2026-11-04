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

## 三张图

| 文件 | 内容 |
|---|---|
| `seal_dataset_statistics.png` / `.pdf` | **(a)(b) 双栏合并图，可直接进论文** |
| `seal_target_number_distribution.png` / `.pdf` | 单图 (a)：横轴每图目标个数 |
| `seal_target_size_distribution.png` / `.pdf` | 单图 (b)：横轴目标大小（像素） |

PNG 为 300 dpi 位图；PDF 为矢量图，字体以 TrueType 子集嵌入（`pdf.fonttype=42`，不是部分期刊拒收的 Type 3），
可直接用于 LaTeX `\includegraphics`。

两栏都以**占比**为纵轴、右轴给**累计占比**（Fig. 4(b) 的做法）：

- (a) 纵轴 = 该目标个数的图像数 / 3,983（%）；右轴 = 累计图像占比。
- (b) 纵轴 = 该尺寸区间的目标数 / 12,554（%）；右轴 = 累计目标占比。
- 横轴分箱：(a) 1–10 精确值，第 11 根柱是 `≥11`（190 张，4.8%），与前面留一个空档，避免把长尾摊成一片矮柱；
  (b) 5 像素一格、0–60；`>60` 的 30 个目标（0.24%，最大 114 px）不画柱，在左上角注明，
  轴与分箱都不被长尾扭曲，累计曲线止于 99.76%。
- 版式按**印刷尺寸**设计：合并图 7.8 × 2.75 in（双栏论文的整栏宽），字号 7.5–8.5 pt，
  PDF 里的 pt 就是最终 pt；单图 3.8 × 2.75 in。合并图用顶部居中的共享图例。
- 配色与细节：柱 `#A7CB55` 填充 + `#5B6B3A` 细边（0.6 pt）；累计曲线 `#C0492F`、1.0 pt、白边小圆点；
  仅左/下轴 + 淡横向网格（`#E8E8E8`，0.5 pt）；轴标签与刻度用深灰 `#333/#4A4A4A`；
  字体 **Microsoft YaHei**，`bbox_inches='tight'`。

图注可用：

> Fig. X. Statistics of the Seal dataset. (a) Target number per image and (b) target size
> distribution. Bars: percentage of images / targets; red lines (right axis): cumulative percentage.

## 复算

统计在服务器上只读 GT 计算（本地只有 test 的 GT），绘图在本地：

```bash
# 服务器（任意 python3 + opencv）
ssh RSML-3 "python /share_datasets/Seal_Model_Reproduction/tools/compute_seal_dataset_stats.py" \
  > Ours/dataset_stats/seal_dataset_stats.json

# 本地
python tools/plot_seal_dataset_stats.py                 # 全部 3,983 张，输出 3 张图
python tools/plot_seal_dataset_stats.py --split test        # 只看 test
```

`seal_dataset_stats.json` 保存每个划分与全部的逐值直方图（`per_image`、`area_hist`）
及面积汇总，重绘不依赖原始 GT。
