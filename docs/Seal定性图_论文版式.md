# Seal 定性图（论文版式，对齐 PDDiff Fig. 8/9）

## 一、PDDiff 的 Fig. 8/9 到底怎么画的

来源：`docs/PDDiff.pdf` 第 9 页（`tmp/pddiff.txt` 已逐字核对为该 PDF 的文本，66211 字符，SHA 一致）。我按 300–1000 dpi 把两个图拆开看过，结构如下（两张图完全同构）：

| 项目 | PDDiff Fig. 8/9 的做法 |
|---|---|
| 列 | 7 列：`Image`、`Mask`、`SCTrans`、`IRSTD-Diff`、`SPKAN`、`HDNet`、`Ours` |
| 每个样例 | 2 行：**第 1 行 2D 面板**（输入 / GT / 各方法二值预测），**第 2 行 3D 响应面**（同一列对应的响应图） |
| 每图样例数 | 2 个样例（Fig. 8 两个海天线场景，Fig. 9 两个复杂背景场景） |
| 表头 | 彩色条：`Image` 橙、`Mask` 浅米、基线方法浅蓝、`Ours` 深蓝；白字/黑字加粗；只在整图顶部出现一次 |
| 2D 面板 | 黑底白目标（GT/预测都按 0/255 显示）；**红色矩形**框出目标区域；右下角一个**红框放大插图**（同一区域 6–8 倍、最近邻放大） |
| Image 面板 | 灰度输入原图，不加红框、不加插图 |
| 3D 响应面 | matplotlib `plot_surface`，`jet` 配色；X/Y 为裁剪块坐标（0–300 左右），Z 为该图的响应值（0–255，按面板各自上限）；`elev≈26, azim≈-64`；白底、浅灰网格、很细的刻度字 |
| Image/Mask 列也有 3D | 是：Image 列是输入强度面（起伏的彩色面），Mask 列是 0/255 的稀疏尖峰面；其余列是各方法的响应面 |
| 结论性差别 | 正文原话：方法只给出不完整响应/背景干扰，Ours "concentrated target peaks with suppressed background responses"——即 3D 面看峰是否集中、背景是否压平 |

补充：PDDiff 官方仓库 `https://github.com/jzw0707/PDDiff` 目前 **404/私有**（`api.github.com/repos/jzw0707/PDDiff` 返回 404，`git clone` 要求认证），论文里写的是 "The code will be published at …"。所以这张图没有官方脚本可抄，本仓库的复现是**按图反推版式**，参数集中在 `tools/build_paper_qualitative.py` 顶部常量里。

## 二、我们怎么复现（已实现）

脚本：`tools/build_paper_qualitative.py`（服务器 `/share_datasets/Seal_Model_Reproduction/tools/`）。

与 PDDiff 的对应关系与 Seal 适配：

| 项目 | 我们的设置 |
|---|---|
| 列 | `Image`、`Mask` + 8 个对比方法 + `Ours`（11 列，`--fig-width 9.6`、`--header-font 7.5`） |
| 输入 Image | Seal 热红外 `T`（`--image T`，可切 `RGB`）。**仅显示用**做 0.5–99.5 百分位重拉伸：Seal 单张热图被一个热点钉住 max，直接显示接近全黑 |
| 二值面板 | `pred >= 128`（概率 > 0.5），与定量表、`AS1000Metrics` 完全同口径 |
| 红框 | 由 **GT 包围框**外扩得到（所有方法共用同一个框，`--roi-min 64 px`、`--roi-margin 24 px`），保证各列可比；Seal 目标只有 3–5 px 宽，64 px 的框约等于目标的 10–20 倍视野 |
| 放大插图 | 右下角、红框、`ZOOM=10`、最近邻，最大占面板 46%（`full` 版式） |
| 3D 响应面 | 取目标中心 `--crop3d 192` 的方形块，`elev=26, azim=-64`、Z 按面板各自上限（`--z-mode per-panel`；`shared` 可强制 0–255）。**必须用概率图**：二值图是 0/255 阶跃，画出来只有一根尖刺；`--surf-maps` 指定 `results/<M>/Seal/fullsup_test_maps`。Image 列用 `gray` + `sigma=5`（热图纹理在 `jet` 下太花），其余列 `jet` + `sigma=2` |
| 3D 坐标/网格（v2） | 三个坐标面画浅灰网格（`_axinfo['grid']` 0.86 灰 / 0.22 pt），X/Y/Z 带整数刻度（`MaxNLocator(nbins=5, integer=True)` → 0/50/100/150 这种好刻度）与轴名，字号 2.1 pt；轴线压到 0.25 pt、0.45 灰，坐标面边框 0.3 pt、0.85 灰——**没有黑色粗轴线**；3D 盒子 `zoom=0.78`，保证刻度字落在本方格内（否则被右邻面板白底盖掉） |
| 字体 | 默认 `--font "Microsoft YaHei,Noto Sans CJK JP,DejaVu Sans"`。服务器没有雅黑，matplotlib 认领 `NotoSansCJK-*.ttc`（`fontconfig` 装好的 `.ttc` matplotlib 默认看不到，脚本里显式 `addfont`）；在装过雅黑的 Windows/macOS 上重跑即自动用雅黑 |
| 面板间距（v2） | `wspace=0.004`、`hspace=0.015`，2D 面板之间只剩 1–2 px 白缝（PDDiff 就是贴着的）；2D 行高度按图像真实比例（`img_h = cell_w×417/622`，ROI 版为正方形），不再有上下白边 |
| 块间距 | 每个样例块之间插 0.10 in 白条（`gap_h`），否则相邻样例的 3D 行和下一块的 2D 行会粘在一起 |
| 输出 | `<tag>_qualitative.png`（600 dpi）+ `<tag>_qualitative.pdf`（矢量）+ `<tag>_meta.json`（列名、样例、阈值、参数） |

两种 2D 行版式（其余部分完全相同）：

| 版式 | 2D 行 | 适用 |
|---|---|---|
| `full`（PDDiff 原版） | 整幅 622×417 二值图 + 红框 + 右下角放大插图 | 与 PDDiff Fig. 8 逐项对应，适合附录/完整版 |
| `roi`（`--roi-panels`） | 每个面板直接就是红框区域，最近邻放大到 384 px（约 6 倍） | **Seal 推荐**：目标只有 3–5 px，整幅图里几乎看不见，只有放大插图能看清；roi 版把所有方法都放到同一放大倍率下比，漏检/虚警一眼可辨 |

样例：`full` 版沿用 `docs/Seal定性对比样例.json`（按 GT 面积分位挑，不按分数挑）；正式图改用 `docs/qualitative_rank_top200.json` 里用户挑选的排名样例（见第六节）。

试跑预览（旧 Ours 占位，仅验证版式）：[`docs/assets/seal_qualitative_paper_dryrun.png`](assets/seal_qualitative_paper_dryrun.png)

## 三、拿到 Ours 结果后怎么接

只要拿到 1,983 张与 `Seal_data/test/GT` 同名的 8-bit PNG（概率图或已二值化的 0/255 都可以），一步接入：

```bash
cd /share_datasets/Seal_Model_Reproduction

# 1) 如果给的是概率图/二值图，先落到统一目录（阈值 >=128 与全项目一致）
python tools/make_binary_maps.py \
  --pred /path/to/ours_maps --out results/Ours_new/Seal/test_binary

# 2) 出论文版式图（把 Ours 换成新目录即可）
python tools/build_paper_qualitative.py \
  --samples docs/Seal定性对比样例.json --num 2 --image T \
  --methods HDNet=results/HDNet/Seal/test_binary \
            L2SKNet=results/L2SKNet/Seal/test_binary \
            BGM=results/BGM/Seal/test_binary \
            DURD=results/DURD/Seal/test_binary \
            RPCANet=results/RPCANet/Seal/test_binary \
  --ours Ours=results/Ours_new/Seal/test_binary \
  --out Ours/qualitative/v6_20261105 --tag seal_final
```

注意事项：

- 文件名必须与 GT 同名（`1.png … 3983.png` 这类）；脚本会逐个校验尺寸，不一致直接报错。
- 若 Ours 输出是 `T` 之外的模态（RGB-T 融合、或多尺度融合），`Image` 列仍建议放 `T`，保证与 12 个单模态基线的输入列一致；如需展示 RGB，可加 `--image RGB` 单独出一版。
- 想换样例：`--ids 3090 3676` 直接指定；或改 `docs/Seal定性对比样例.json`。
- 12 个方法全放一列会有 14 列、每列 0.5 in，字太小；论文主图建议 5 个基线 + Ours，全套可放附录（把 `--methods` 全部列出即可）。

## 四、推荐的两组列（论文主图用）

| 组名 | 列 | 理由 |
|---|---|---|
| 近期方法 | HDNet(25) / L2SKNet(25) / BGM(25) / DURD(26) / RPCANet++(26) + Ours | 全是 2025–2026，时间线最贴近投稿 |
| 按指标 | BGM(25, IoU 0.307) / RPCANet(24, 0.306) / MSHNet(24, 0.292) / L2SKNet(25, 0.301) / HDNet(25, 0.273) + Ours | 表 21–32 行里 IoU/F1 最高的几个 |

## 五、已知差异与待办

1. PDDiff 的 3D 面看起来更"饱满"，是因为他们的响应图有较宽的软响应；Seal 的滑窗推理输出很干脆（目标处直接到 250+、背景接近 0），3D 上表现为细尖峰 + 平坦背景——这本身就是"响应集中"的证据，`smooth=2.0` 只是显示抗锯齿，不改数值。
2. 版式里的红框按 GT 画；如果审稿人要求框出"预测"的漏检/虚警，可再加一组红/蓝双色框版本（`compose_panel` 里留了 `with_box` 参数，扩展即可）。
3. 11 列 × 11 样例在 9.6 in 宽下每列 0.87 in。若论文按 7.2 in 双栏整宽排版，列宽缩到 0.65 in、表头 7.5 pt 会掉到 5.6 pt——**主图建议用 `seal_pick6_roi` + `seal_pick5_roi` 两张**（各 6 / 5 个样例），11 张一起的 `seal_pick11*` 只适合附录整页或汇报 PPT。

## 六、2026-11 最终图（用户挑选的 11 个样例）

样例来源：`docs/qualitative_rank_top200.json`（`tools/rank_qualitative_samples.py` 按 `score = F1(Ours) − mean_k F1(baseline_k)` 排序前 200，并在 `docs/qualitative_review_top200.html` 里逐图带缩略图复核）。用户从中挑选排名 **1,2,5,6,8,11,13,15,19,21,29**：

| 排名 | ID | GT 面积 (px) | GT 尺寸 | Ours 预测 |
|---:|---|---:|---|---|
| 1 | 2656 | 12 | 5×4 | 12 px，1 连通域 |
| 2 | 2735 | 10 | 3×4 | 8 px |
| 5 | 2185 | 11 | 4×4 | 13 px |
| 6 | 3899 | 10 | 3×5 | 9 px（8 个基线全部漏检，SAM3 有响应） |
| 8 | 3896 | 11 | 5×4 | 7 px |
| 11 | 2712 | 11 | 4×4 | 11 px |
| 13 | 3398 | 26 | 9×15 | 18 px，2 连通域 |
| 15 | 2688 | 45 | 49×57 | 36 px，3 连通域 |
| 19 | 2120 | 103 | 19×97 | 92 px，3 连通域 |
| 21 | 2894 | 24 | 98×72（多目标） | 20 px，3 连通域 |
| 29 | 2888 | 9 | 4×4 | 12 px |

（`2688/2120/2894` 是多目标或细长目标，视野被 `--roi-min 64` 自动放大到 98–138 px。）

构建：`bash tools/make_paper_qualitative_pick.sh all Ours/qualitative/v11_20261105`（第二个参数是输出目录，默认 `Ours/qualitative/latest`；**每次改版都给新目录**，别覆盖旧版）。`full`/`roi` 二选一也行。

### 版本

| 版本 | 服务器目录 | 本地目录 | 说明 |
|---|---|---|---|
| v1 | `Ours/qualitative/v1_20261104_11col/` | 已删除 | 首版：3D 无刻度、有黑色粗轴线、列间距 0.02、Image 列 `gray+sigma3` |
| v2 | `Ours/qualitative/v2_20261105/` | 已删除 | 列间距压到 0.004、2D 行按真实比例、3D 加网格+整数刻度+轴名、去掉黑粗轴线、3D 盒子 `zoom=0.78` |
| v3 | `Ours/qualitative/v3_20261105/` | 已删除 | 3D 刻度 1.8 pt、`zoom=0.85`；新增单图拆分 `panels_roi/` 与 Visio 版 |
| v4 | `Ours/qualitative/v4_20261105/` | 已删除 | 3D 盒子铺满格子：每个 3D 先单独渲染（0.873×0.733 in @1200 dpi，与格子同尺寸同字号）→ 裁白边（26 px）→ `aspect='auto'` 贴回；拼图里的 3D 行因此变成位图，PDF 从 5–10 MB 降到 1–2.3 MB |
| v5 | `Ours/qualitative/v5_20261105/` | 已删除 | 3D 更满（裁边 26→6 px）+ 峰更高更亮：平滑 σ 2.0→0.9、`vmax`=平滑后峰值、Z 轴统一 0–255、盒子 z 比例 0.55→1.00、自定义 `seal3d` 色带（深蓝底→青→黄→亮红）+ `PowerNorm(0.45)`；新增 `--diag` 诊断面板（TP 白 / FP 红 / FN 绿 + 红框），Visio 改用诊断面板 |/n| v6（当前） | `Ours/qualitative/v6_20261105/` | `Ours/qualitative/v6_20261105/` | 诊断面板的红框放大：从「GT 外扩 8 px」（384 面板里约 40 px）改成**至少占面板 36%（约 138 px）、3 px 线**，居中套在 GT 上；其余与 v5 相同 |
| v6 | 已删除 | 已删除 | 诊断面板的红框放大：从「GT 外扩 8 px」（384 面板里约 40 px、2 px 线）改成**至少占面板 36%（约 138 px）、3 px 线**，居中套在 GT 上；其余与 v5 相同 |
| v7 | 已删除 | 已删除 | 3D 网格/刻度清晰化（网格 0.62 灰 0.45 pt、刻度 2.2 pt 近黑、每轴 3 个刻度、去掉 X/Y/Z 轴名避免与数字相撞）；Image 列 3D 改彩色（与 PDDiff 一致，线性映射）；诊断面板多目标时**每个连通域一个红框**（单目标 36%、多目标 0.45/n）；裁边 6→14 px |
| v8 | 已删除 | 已删除 | **Image 列改成同一块 ROI 的 6 倍放大**（原来整幅 622×417），诊断模式下套与其它列完全相同的 GT 定位框——之前 Image 的红框是「ROI 窗口」、其它列是「GT 定位框」，两者不对应；顺带 Image 列从 0.85×0.57 in 的小图变成与其它列同尺寸 |
| v9 | 已删除 | 已删除 | **去掉黑边**：ROI 改成「固定边长正方形、贴边整体内移」，`side = min(max(min_box, span+2·margin), H, W)`，窗口永远完整落在图像内，不再出现 64×43 这种被裁后补黑的非方形；贴边目标的 GT 因此不在正中心而靠近对应边框（红框跟 GT 走，位置仍正确） |
| v10 | 已删除 | 已删除 | **红框只留一个**：Image 列回到整幅 622×417（压缩成正方形）并画唯一一个正方形红框，表示后面各列是这块区域放大出来的；Mask…Ours 不再画任何红框（保留 TP 白/FP 红/FN 绿）。红框边长按 ROI 面积换算（`ROI×sqrt(sx·sy)`），64 px ROI → 约 12.5% 面板。另提供 `--image-mode crop417`（417×417 原比例窗口，不变形、框即真实 ROI） |
| v11（当前） | `Ours/qualitative/v11_20261105/` | `Ours/qualitative/v11_20261105/` | **Ours 换成 1105 最新预测**（`Ours/test_maps` 由旧预测改名 `test_maps_20261105_prev`，新预测解到 `test_maps`）；版式与 v10 全同。11 个样例在新预测下 F1 0.55–0.87、目标全检出，9 个胜过全部基线、2 个仅比 SAM3 低 0.01–0.02，因此沿用原 11 张；**Image 列红框尺寸统一**：边长固定 `64×sqrt(384/622×384/417) ≈ 48 px`（含边框实测 51–54 px），不随 GT 跨度变化，大目标（2688/2120/2894）的框比真实 ROI 小，红框只作定位符号 |

服务器保留全部历史版本；本地只留最新一版，**整个 `Ours/qualitative/` 不进 git**（单版 100 MB 量级，`.gitignore` 已整目录排除）。

v11 产物（600 dpi PNG + 矢量 PDF + meta）：

| 文件 | 内容 | 尺寸 |
|---|---|---|
| `seal_pick11_qualitative.png/pdf` | 11 样例 × 11 列，`full` 版式（整幅 + 红框 + 插图） | 5760×10086 px / 9.6×16.8 in |
| `seal_pick6_qualitative.*` / `seal_pick5_qualitative.*` | 同样例拆成 6 + 5 两张 | 5760×5569 / 5760×4666 px |
| `seal_pick11_roi_qualitative.*` | 11 样例，`roi` 版式（面板 = 6 倍放大 ROI） | 5760×11408 px / 9.6×19.0 in |
| `seal_pick6_roi_*` / `seal_pick5_roi_*` | 同上拆两张 | 5760×6290 / 5760×5267 px |
| `seal_diag_roi_qualitative.*` | 诊断版拼图（TP 白 / FP 红 / FN 绿 + 红框） | 5760×11408 px |
| `panels_roi/…` / `panels_diag/…` | 单图拆分各 242 张：`<nn>_<id>/2d|3d/<col>.png` | 2D 384×384；3D ~923×806（已裁白边） |
| `seal_qualitative_11col.vsdx` | Visio 可编辑版（诊断面板）：11 页，34 个独立形状/页 | 页面 18.3×4.3 in |

单图拆分的 3D 图就是拼图里贴的那张（同一次渲染、缓存目录 `<out>/_3d_cache/`），所以 Visio 里看到的效果与拼图完全一致。

3D 的 Z 轴用 `--z-mode shared` 固定 0–255：所有列都是同一量纲的概率图，跨列可比（PDDiff 也是这么做的）；峰高=真实响应强度，颜色用 `PowerNorm(gamma=0.45)` 让中等响应也偏暖色，避免"响应弱就看不出峰"。

Visio 版由 [tools/make_visio_qualitative.ps1](../tools/make_visio_qualitative.ps1) 用 Visio COM 生成（仅 Windows + 已装 Visio 可用）：表头是带填充/文字/雅黑的矩形，图片是嵌入的图片形状，页脚一行说明——都能单独选中改格式。

仓库内预览（1800 px 宽，仅用于查看版式，已同步为 v11）：[`seal_qualitative_pick6_roi.png`](assets/seal_qualitative_pick6_roi.png)、[`seal_qualitative_pick5_roi.png`](assets/seal_qualitative_pick5_roi.png)、[`seal_qualitative_pick11_full.png`](assets/seal_qualitative_pick11_full.png)。

从图上直接能读到的事实（对照表 21～32 行）：

- `3899`：8 个基线全部空预测，只有 SAM3 和 Ours 出目标——但 SAM3 的斑块明显大于 GT（GT 10 px，SAM3 一坨），Ours 的斑块与 GT 同量级。
- `2656`：`RPCANet++` 在框内画出一道弧（虚警），`RPCANet`/`BGM` 有零散噪点，`DNA-Net`/`MSHNet`/`HDNet`/`L2SKNet` 全空，Ours 单点命中。
- `2888`：9 个像素的极小目标，只有 Ours 和 SAM3 命中，两者都偏大 1–3 px。
- 与定量表一致：SAM3 检出率高但斑块偏大（Fa 与像素级 F1 都不如 Ours），12 个基线在 3–5 px 目标上大量漏检。定性图只能说明"命中/漏检"的差别，不能替代表里的 IoU/F1/Pd/Fa。
