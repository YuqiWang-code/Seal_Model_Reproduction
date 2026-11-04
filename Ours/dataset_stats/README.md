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

## 两张图

| 文件 | 内容 |
|---|---|
| `seal_target_number_distribution.png` | 横轴：每图目标个数；纵轴：图像数（样本数） |
| `seal_target_size_distribution.png` | 横轴：目标大小（像素）；纵轴：mask 个数（目标数） |

- 横轴分箱：目标个数用 1–10 精确值 + `11-15` / `16-20` / `>20` 三组；目标大小用 5 像素一格（0–60），
  末端 `>60` 单独一根柱（全部数据里只有 30 个目标，占 0.24%，最大 114 px），避免长尾把主体压扁。
- 样式：单色柱 + 黑边，仅保留左/下坐标轴、无网格，柱顶标数值；字体 **Microsoft YaHei**，
  300 dpi，`bbox_inches='tight'`。
- 两张图的柱颜色统一为 `#8CBF3F`。

## 复算

统计在服务器上只读 GT 计算（本地只有 test 的 GT），绘图在本地：

```bash
# 服务器（任意 python3 + opencv）
ssh RSML-3 "python /share_datasets/Seal_Model_Reproduction/tools/compute_seal_dataset_stats.py" \
  > Ours/dataset_stats/seal_dataset_stats.json

# 本地
python tools/plot_seal_dataset_stats.py                     # 全部 3,983 张
python tools/plot_seal_dataset_stats.py --split test        # 只看 test
```

`seal_dataset_stats.json` 保存每个划分与全部的逐值直方图（`per_image`、`area_hist`）
及面积汇总，重绘不依赖原始 GT。
