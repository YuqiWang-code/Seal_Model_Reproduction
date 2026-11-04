# Seal 全监督基线的复杂度指标：Params / GFLOPs / FPS

`docs/Seal量化SOTA表.xlsx` 第 13–15 列（`Params↓`、`GFLOPs↓`、`FPS↑`）的测量口径、实测值，以及与各方法原论文/官方仓库公布值的对照。测量对象是第 21–32 行的 12 个全监督 IRSTD 基线；第 3–20 行弱监督/跨域方法不在此次测量范围内。

## 测量协议

| 项目 | 口径 |
|---|---|
| 输入 | `1×C×256×256`，FP32，batch 1；单通道方法 C=1，三通道方法（DNA-Net / MSHNet / UIU-Net / HDNet / ILNet）C=3，与各自 `train_seal.py` 的输入一致 |
| Params | 网络全部 `nn.Parameter` 之和（`sum(p.numel() for p in net.parameters())`），不含 buffer |
| GFLOPs | `thop.profile` 的 **MACs** 计数（乘加数），单位 1e9。这是 IRSTD 论文里写 "FLOPs / GFLOPs" 时的实际口径（见下节与 LPCNet 表的一致性） |
| FPS | `1000 / 单次前向延迟中位数(ms)`。一轮 = 50 次 warm-up + 100 次计时取中位数，共跑 3 轮，取三轮中位数里最小（最好）的一轮；每轮开始前先做 0.4 s 矩阵乘负载，整体开始前做 2 s，把 GPU 时钟从空闲的 180 MHz 拉满（不预热会把 HDNet 误测成 155 ms） |
| 硬件 | 单卡 RTX 5090（服务器 `cuda:1`），PyTorch + thop，cudnn benchmark 开启 |
| 排除 | 不计数据读取、归一化、滑窗拼接、sigmoid 与 PNG 写盘 |
| 脚本 | [`tools/measure_complexity_one.py`](../tools/measure_complexity_one.py)，批量入口 `tmp/fullsup4/measure_all.sh`，原始 JSON 在服务器 `results/complexity/<kind>.json` |

评测一张 Seal test 图的真实成本要乘滑窗数：`622×417` 原生分辨率、窗口 256、步长 128 时纵向 3 个、横向 4 个起点，共 **12 个窗口**（[`seal_infer.py`](../DRPCA-Net-main/seal_infer.py) 的 `_windows` 会补齐末尾窗口）。所以单图成本约 `12×GFLOPs`，单图吞吐约 `FPS/12`（批量 8 时略好于线性折算）。

## 实测结果

| 方法 | 会议 | Params↓ (M) | GFLOPs↓ | FPS↑ | 单次前向 (ms) |
|---|---|---:|---:|---:|---:|
| DNA-Net | TIP22 | 4.697 | 14.282 | 64.22 | 15.571 |
| MSHNet | CVPR24 | 4.066 | 6.112 | 152.32 | 6.565 |
| UIU-Net | TIP23 | 50.541 | 54.501 | 118.39 | 8.446 |
| SCTransNet | TGRS24 | 11.326 | 10.119 | 109.15 | 9.161 |
| RPCANet | WACV24 | 0.680 | 44.569 | 211.29 | 4.733 |
| HDNet | TGRS25 | 3.676 | 5.697 | 85.54 | 11.691 |
| ILNet（L 版） | TAES25 | 4.040 | 21.421 | 108.39 | 9.226 |
| RPCANet++（S6） | TPAMI26 | 2.962 | 190.669 | 80.42 | 12.434 |
| L2SKNet | TGRS25 | 0.899 | 6.895 | 332.16 | 3.011 |
| BGM | TGRS25 | 4.076 | 6.773 | 212.14 | 4.714 |
| DRPCA-Net | TGRS25 | 1.169 | 73.837 | 11.23 | 89.044 |
| DURD | TGRS26 | 2.466 | 52.847 | 125.34 | 7.978 |

排序读法：参数最少的是 RPCANet（0.680 M）与 L2SKNet（0.899 M）；计算量最大的是 RPCANet++（190.7 G，6 阶段展开）与 DRPCA-Net（73.8 G）；单次延迟最低的是 L2SKNet（3.01 ms），最高的是 DRPCA-Net（89.0 ms）。参数少不等于延迟低——RPCANet 只有 0.68 M 却要 44.6 G MACs，因为它把 RPCA 展开成 6 个阶段、每阶段在全分辨率上做多组卷积。

## 与原论文 / 官方仓库公布值的对照

只填原论文或作者仓库真正给出过的数字；同一项有多个来源时全部列出，`—` 表示该来源未给出。

| 方法 | 我们 Params (M) | 公开 Params (M) | 我们 GFLOPs | 公开 GFLOPs | 公开速度 | 来源 |
|---|---:|---|---:|---|---|---|
| DNA-Net | 4.697 | 4.697 / 4.69 / 4.70 | 14.282 | 13.9 / 14.26 | 22.64 ms（3090，256²） | DRPCA-Net README 对比表；LPCNet Tab. I；PDDiff Tab. I |
| MSHNet | 4.066 | 4.066 / 4.06 / 4.07 | 6.112 | 6.01 / 19.27 | 13.52 ms（3090） | 同上 |
| UIU-Net | 50.541 | 50.54 / 50.5 / 50.54 | 54.501 | 54.2 / 54.43 | 27.50 ms（3090） | 同上 |
| SCTransNet | 11.326 | 11.3 | 10.119 | 10.0 / 20.24 | 31.42 ms（3090） | LPCNet Tab. I；PDDiff Tab. I |
| RPCANet | 0.680 | 0.680 / 0.68 | 44.569 | 44.3 | 0.0217 s（GPU，46.1 FPS） | RPCANet 仓库 README；LPCNet Tab. I |
| HDNet | 3.676 | 3.68 | 5.697 | 11.20 | 16.70 FPS（PDDiff 管线） | PDDiff Tab. I |
| ILNet | 4.040 | 0.046（ILNet-S） | 21.421 | 2.19（ILNet-S @512²） | Jetson AGX Xavier 上给 FPS 曲线 | ILNet 论文 |
| RPCANet++ | 2.962 | 2.915（S6，thop 口径） | 190.669 | 190 | 0.0471 s（21.2 FPS） | RPCANet 仓库 README；LPCNet Tab. I |
| L2SKNet | 0.899 | 0.899 / 1.07 | 6.895 | 6.895 / 5.92 | 9.38 ms（3090） | L2SKNet `t_models.py`；DRPCA-Net README；LPCNet Tab. I |
| BGM | 4.076 | — | 6.773 | — | — | 论文与仓库均未给出 |
| DRPCA-Net | 1.169 | 1.169 | 73.837 | — | — | DRPCA-Net 仓库 README |
| DURD | 2.466 | — | 52.847 | — | — | 论文与仓库均未给出 |

`LPCNet Tab. I` 指 arXiv:2607.04603v3 第 8 页的对比表（RTX 3090，256×256，统一用 thop 计 Params(M)/Flops(G)/GPU ms），是本批最新、口径最接近我们的一次第三方复测。

### Params：有公开值的 9 个全部对上

- **逐位相同**：DNA-Net 4.697（DRPCA-Net 对比表）、UIU-Net 50.54（同表）、MSHNet 4.066（同表）、RPCANet 0.680（RPCANet 仓库 + 同表）、L²SKNet 0.899（作者仓库 `t_models.py` 打印 0.898913 M，DRPCA-Net 对比表也是 0.899）、DRPCA-Net 1.169（作者仓库）。
- **四舍五入后相同**：DNA-Net 4.69 / LPCNet、UIU-Net 50.5 / LPCNet、MSHNet 4.06 / LPCNet、SCTransNet 11.3 / LPCNet、RPCANet 0.68 / LPCNet、RPCANet++ 2.96 / LPCNet（我们 2.961582，差 0.05 %，见下条）。
- **RPCANet++ 的 2.915 vs 2.962（+1.6 %）是计数约定差，不是模型差**：RPCANet 仓库 README 的 2.915 M 由 `t_params.py` 的 `thop.profile` 得到，而 thop 只累加 conv/linear/bn 等它注册过的模块的参数，**不含 `nn.LSTM`**。同一网络实测：thop 口径 2.915336 M、全部参数 2.961582 M，差值 0.046 M 恰是 LSTM 部分。本表统一用"全部 `nn.Parameter`"，与 LPCNet 的 2.96 一致。
- **L2SKNet 的 1.07 是第三方差异**：作者仓库自己的 `t_models.py` 在 256² 上打印 0.898913 M / 6.894535 G，与我们的 0.899 / 6.895 逐位一致；LPCNet 表的 1.07 M / 5.92 G 应来自另一份 L²SKNet 实现或不同宽度配置。
- **ILNet 的 0.046 M 是 S 版，我们训的是 L 版**：论文原文写"最小的 ILNet … 2.19 GFLOPs、0.046 M 参数"。用作者代码实测三个版本：`ILNet_S` 0.045980 M、`ILNet_M` 1.581631 M、`ILNet_L` 4.039792 M；`ILNet_S` 在 512² 下 thop MACs 为 2.2146 G，与论文 2.19 G 差 1.1 %。即我们的计数方法是对的，表里的 4.040 M 对应官方仓库默认的 `--mode L`（我们用 L 版训练，与 `train_seal.py` 一致），不是论文摘要里那个 S 版数字。
- **BGM / DURD 无数可比**：BGM 用 `BasicUNet_plus` 主干（论文以掩码策略为主，未列表参数量），DURD 是 TGRS 2026 新文，仓库 README 无参数表，公开检索也未拿到表格；这两行只能是我们按官方代码的实测值。

### GFLOPs：论文里的 "FLOPs" 就是 MACs

以 LPCNet 表为准逐项对照，我们的 MACs 与它的 "Flops (G)" 基本重合：

| 方法 | 我们 | LPCNet | 相对差 |
|---|---:|---:|---:|
| DNA-Net | 14.282 | 13.9 | +2.7 % |
| MSHNet | 6.112 | 6.01 | +1.7 % |
| UIU-Net | 54.501 | 54.2 | +0.6 % |
| SCTransNet | 10.119 | 10.0 | +1.2 % |
| RPCANet | 44.569 | 44.3 | +0.6 % |
| RPCANet++ | 190.669 | 190 | +0.4 % |
| L2SKNet | 6.895 | 5.92 | +16.5 %（同上的实现差异） |

残余几个百分点的差异来自输入尺寸细节（对齐/补齐、是否含 `AdaptiveAvgPool` 分支）与 thop 版本，量级上完全一致。这说明这些论文报的 "FLOPs/GFLOPs" 是**乘加数**，不是 2×MACs 的 FLOPs 定义。

PDDiff 的表则不自洽：它给 SCTransNet 20.24 G、HDNet 11.20 G，正好是我们 MACs 的 2.00×（= 按 FLOPs=2×MACs 计），但 MSHNet 给 19.27 G，是我们 6.112 G 的 3.15×，无法用 2× 解释；同时它给的 Params（4.70 / 50.54 / 11.33 / 3.68 / 4.07）与我们逐个吻合，说明模型本身相同。因此本表统一采用 MACs 口径，并对 PDDiff 的 GFLOPs 只作参考、不作比对基准。

### FPS：跨论文不可比，只能看量级与相对关系

各来源的测速条件差异很大——PDDiff 的 FPS 明显是含整图滑窗与预处理的管线口径（DNANet 9.36 FPS ≈ 107 ms/张，比同一模型在 3090 上单窗 22.64 ms 慢 4.7×）；ILNet 论文的 FPS 曲线是在 **Jetson AGX Xavier**（无 TensorRT）上测的；RPCANet 系仓库给的是"每张图 CPU/GPU 秒数"。把能换算的统一换成毫秒后与我们的 5090 单窗延迟对比：

| 方法 | 对方 | 换算 | 我们（5090） | 我们/对方 |
|---|---|---:|---:|---:|
| RPCANet | 0.0217 s（RPCANet 仓库） | 21.7 ms | 4.73 ms | 4.6× |
| RPCANet++ S6 | 0.0471 s（RPCANet 仓库） | 47.1 ms | 12.43 ms | 3.8× |
| L2SKNet | 9.38 ms（3090） | 9.38 ms | 3.01 ms | 3.1× |
| MSHNet | 13.52 ms（3090） | 13.52 ms | 6.57 ms | 2.1× |
| UIU-Net | 27.50 ms（3090） | 27.50 ms | 8.45 ms | 3.3× |
| SCTransNet | 31.42 ms（3090） | 31.42 ms | 9.16 ms | 3.4× |
| RPCANet++ | 43.61 ms（3090） | 43.61 ms | 12.43 ms | 3.5× |
| DNA-Net | 22.64 ms（3090） | 22.64 ms | 15.57 ms | 1.45× |

除 DNA-Net 外，5090 在 256² 单窗上比 3090 快 2.1–4.6×，与两代卡的实测差距一致；RPCANet 系仓库的 21.7 / 47.1 ms 与 3090 上的 15.4 / 43.6 ms 本身也吻合，说明其换算到我们这边同样落在 3.8–4.6×，这十几个数字互相印证，没有"太离谱"的项。DNA-Net 的 1.45× 是唯一明显偏小的：单独复测它的 100 次计时分布很紧（min 14.66、p50 15.75、p90 15.83 ms），15.6 ms 是稳定值而非噪声——它只有 14.3 G MACs，延迟主要由大量小卷积的 kernel 启动决定，新卡的算力提升摊不到这部分开销上，所以其 FPS（64.2）偏保守但测量无误。

### DRPCA-Net 的 11.23 FPS 不是测量噪声

它是 12 个方法里唯一"MACs 高、吞吐又低"的：73.8 G MACs 却要 89 ms，单窗有效算力仅 0.83 TFLOP/s（RPCANet 为 44.6 G / 4.73 ms ≈ 9.4 TFLOP/s）。三条证据说明这是官方实现本身的开销：

1. **对 batch size 不敏感**：同一前向 bs=1/2/4 的中位延迟为 85 / 56 / 65 ms，几乎不随批量增长——说明瓶颈不在 GPU 算力，而在逐层的小 kernel 启动与权重生成。
2. **`torch.profiler` 定位到具体算子**：单次前向 CUDA 时间合计 49.2 ms，其中 **40.2 ms（81.8 %）来自 `at::native::adaptivemaxpool`**，每前向调用 29 次；它出现在 `ChannelAttention` 的 `nn.AdaptiveMaxPool2d(1)`（CBAM 式的通道注意力），ATen 的这个 kernel 在 32 通道特征图上极慢。真正的卷积只占 5 % 左右。
3. **结构上确实重**：6 个展开阶段 × 每阶段的 `LowrankModule` / `DynamicSparseModule` / `DynamicResidualMergeModule`，且 `DynamicSpatialAttention` 用运行时生成的权重做 `F.conv2d(groups=B)`，权重每次前向都要重新生成，cudnn 无法复用最优算法。

本表按官方代码如实记录 11.23 FPS，并在正文中把它标注为"实现开销受限"的方法；若换掉那个 `AdaptiveMaxPool2d(1)`（等价改写为 `amax(dim=(2,3), keepdim=True)`）延迟会大幅下降，但这属于改进实现，不能用于对比公平性，故不做。

## 与表中其他列的配合读法

- 复杂度三列与八项指标同源（同一份官方代码、同一输入尺寸），但**与跨域行（第 3–20 行）不同协议**，不能横向拼比。
- 参数/计算量与 Seal 上的精度没有单调关系：参数最少的 RPCANet（0.680 M）与计算量最大的 RPCANet++（190.7 G）在 Seal test 上的 IoU 是 0.30638 / 0.30095，几乎相同；BGM（4.076 M / 6.773 G）反而是 IoU 最高的 0.30735。12 个方法的 IoU 全部落在 0.19–0.31、F1 0.32–0.47 的窄带内，与 [IoU 边界敏感性](小目标评价指标_定义与代码对照.md#iou-的边界敏感性seal-gt-实测)（1 像素膨胀/腐蚀即 0.3910/0.0826）一致，因此复杂度列的作用是描述"达到同一水平所需的代价"，而不是解释分数高低。
- 写论文时若需要"整图耗时"，用 12 个滑窗折算（GFLOPs×12、FPS÷12）；若需要与论文表对照，用单窗 256² 的数值，并注明硬件。

## 复现

```bash
# 服务器，每个方法目录各跑一次（cwd 必须是方法目录，模型才 import 得到）
cd /share_datasets/Seal_Model_Reproduction/DNANet-main
/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python \
  /share_datasets/Seal_Model_Reproduction/tools/measure_complexity_one.py --kind dnanet --device cuda:1

# 12 个方法批量跑，结果写 results/complexity/<kind>.json
bash /share_datasets/Seal_Model_Reproduction/tmp/fullsup4/measure_all.sh
```

`--kind` 与目录的对应：`dnanet`→`DNANet-main`、`mshnet`→`MSHNet-main`、`uiunet`→`UIU-Net-main`、`sctransnet`→`SCTransNet-main`、`rpcanet`/`rpcanetpp`→`RPCANet-main`、`hdnet`→`HDNet-main`、`ilnet`→`ILNet-main`、`l2sknet`→`L2SKNet-main`、`bgm`→`BGM-main`、`drpcanet`→`DRPCA-Net-main`、`durd`→`DURD-main`。

填表脚本按列标题定位（不依赖列号）：`tmp/fullsup4/fill_complexity_xlsx.py`。
