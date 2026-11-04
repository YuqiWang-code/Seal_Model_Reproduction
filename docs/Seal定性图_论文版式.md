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
| 列 | `Image`、`Mask` + 5 个对比方法 + `Ours`（7 列，和论文同宽 7.2 in，双栏排版正好整宽） |
| 输入 Image | Seal 热红外 `T`（`--image T`，可切 `RGB`）。**仅显示用**做 0.5–99.5 百分位重拉伸：Seal 单张热图被一个热点钉住 max，直接显示接近全黑 |
| 二值面板 | `pred >= 128`（概率 > 0.5），与定量表、`AS1000Metrics` 完全同口径 |
| 红框 | 由 **GT 包围框**外扩得到（所有方法共用同一个框，`MIN_BOX=96 px`、`MARGIN=32 px`），保证各列可比 |
| 放大插图 | 右下角、红框、`ZOOM=8`、最近邻，最大占面板 40% |
| 3D 响应面 | 取目标中心 `--crop3d 192` 的方形块，`jet`、`elev=26, azim=-64`、Z 默认按面板各自上限（`--z-mode per-panel`，与论文一致；`shared` 可强制 0–255 统一量程） |
| 输出 | `<tag>_qualitative.png`（600 dpi）+ `<tag>_qualitative.pdf`（矢量）+ `<tag>_meta.json`（列名、样例、阈值、参数） |

样例选择沿用 `docs/Seal定性对比样例.json`（按 GT 面积分位 + 两个多目标帧，**不按任何方法分数挑**）。

试跑预览（占位 Ours，仅验证版式）：[`docs/assets/seal_qualitative_paper_dryrun.png`](assets/seal_qualitative_paper_dryrun.png)

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
  --out qualitative_paper --tag seal_final
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

1. PDDiff 的 3D 面看起来更"饱满"，是因为他们的响应图有较宽的软响应；Seal 的滑窗推理输出很干脆（目标处直接到 250+、背景接近 0），3D 上表现为细尖峰 + 平坦背景——这本身就是"响应集中"的证据，不需要额外平滑（脚本默认 `smooth=0.8`，仅用于显示抗锯齿）。
2. 版式里的红框按 GT 画；如果审稿人要求框出"预测"的漏检/虚警，可再加一组红/蓝双色框版本（`compose_panel` 里留了 `with_box` 参数，扩展即可）。
3. 目前 `qualitative_paper/dryrun_*` 用的是**旧 Ours 预测**（`Ours/test_maps`）占位，仅用于验证版式；正式图等新的 Ours 结果到位后按第三节命令重出。
