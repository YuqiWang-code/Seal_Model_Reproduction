"""Insert a '## 复杂度' section (before '## 产物与已知问题') in the 12 baseline records."""
import io
import os

DOCS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'docs')
ANCHOR = '## 产物与已知问题'

TEMPLATE = """## 复杂度

| Params↓ (M) | GFLOPs↓ | FPS↑ | 单次前向 (ms) |
|---:|---:|---:|---:|
| {par} | {gf} | {fps} | {ms} |

单窗 256×256、FP32、batch 1、RTX 5090 实测（thop MACs、100 次×3 轮取中位）。Seal test 一张 622×417 需 12 个滑窗，单图成本约 ×12、单图吞吐约 FPS÷12。{note}

"""

DATA = {
    'DNANet': ('4.697', '14.282', '64.22', '15.571',
               '对照：作者对比表与技术报告给 4.697 M，LPCNet 表（RTX 3090）给 4.69 M / 13.9 G，PDDiff 表给 4.70 M / 14.26 G，与实测一致。'),
    'MSHNet': ('4.066', '6.112', '152.32', '6.565',
               '对照：DRPCA-Net 对比表 4.066 M、LPCNet 表 4.06 M / 6.01 G，一致；PDDiff 表的 19.27 G 是它自己的计数口径（约 3.15×MACs），不作比对基准。'),
    'UIU-Net': ('50.541', '54.501', '118.39', '8.446',
               '对照：作者对比表 50.54 M，LPCNet 表 50.5 M / 54.2 G，PDDiff 表 50.54 M / 54.43 G，一致。'),
    'SCTransNet': ('11.326', '10.119', '109.15', '9.161',
               '对照：LPCNet 表 11.3 M / 10.0 G，与实测一致；PDDiff 表的 20.24 G 正好是 2×MACs（FLOPs=2×MACs 的定义差），不作比对基准。'),
    'RPCANet': ('0.680', '44.569', '211.29', '4.733',
               '对照：RPCANet 仓库 README 0.680 M、单图 GPU 0.0217 s，LPCNet 表 0.68 M / 44.3 G，一致。'),
    'HDNet': ('3.676', '5.697', '85.54', '11.691',
               '对照：PDDiff 表 3.68 M，参数一致；其 GFLOPs 11.20 G 是 2×MACs 口径，不作比对基准。'),
    'ILNet': ('4.040', '21.421', '108.39', '9.226',
               '对照：ILNet 论文首页的"0.046 M / 2.19 GFLOPs"指 `ILNet_S`（实测 0.045980 M、512² 下 2.2146 G，与论文吻合），本表训练的是仓库默认 `ILNet_L`，故参数为 4.040 M；论文的 FPS 曲线在 Jetson AGX Xavier 上测得，不可与 5090 直比。'),
    'RPCANet++': ('2.962', '190.669', '80.42', '12.434',
                  '对照：RPCANet 仓库 README 的 S6 为 2.915 M / 单图 GPU 0.0471 s，LPCNet 表 2.96 M / 190 G。2.915 vs 2.962 的 1.6 % 差异是 thop 不计 `nn.LSTM` 参数（thop 口径 2.915336 M、全部参数 2.961582 M），模型相同。'),
    'L2SKNet': ('0.899', '6.895', '332.16', '3.011',
                '对照：本仓库 `t_models.py`（thop、256²）打印 0.898913 M / 6.894535 G，与实测逐位一致；DRPCA-Net 对比表亦为 0.899 M。LPCNet 表的 1.07 M / 5.92 G 应来自另一份实现，不作基准。'),
    'BGM': ('4.076', '6.773', '212.14', '4.714',
            '对照：论文与官方仓库均未给出 Params / GFLOPs / FPS，本行为按官方代码（`BasicUNet_plus` 主干）的实测值。'),
    'DRPCA-Net': ('1.169', '73.837', '11.23', '89.044',
                  '对照：作者对比表给 1.169 M，与实测一致。FPS 偏低是实现开销而非测错：`torch.profiler` 显示单次前向 CUDA 时间的 81.8 % 花在 `ChannelAttention` 的 `nn.AdaptiveMaxPool2d(1)`（每前向 29 次调用），且延迟对 batch size 不敏感，见[复杂度指标说明](Seal复杂度指标_说明.md#drpca-net-的-1123-fps-不是测量噪声)。'),
    'DURD': ('2.466', '52.847', '125.34', '7.978',
             '对照：论文（TGRS 2026）与官方仓库均未给出 Params / GFLOPs / FPS，本行为按官方代码的实测值。'),
}

for method, (par, gf, fps, ms, note) in DATA.items():
    path = os.path.join(DOCS, f'{method}_Seal实验记录.md')
    if not os.path.exists(path):
        # RPCANet++ file name uses '++'
        alt = os.path.join(DOCS, f'{method}_Seal实验记录.md'.replace('RPCANet++', 'RPCANet++'))
        path = alt
    text = open(path, encoding='utf-8').read()
    if '## 复杂度' in text:
        print(f'skip {method}: already has 复杂度')
        continue
    if text.count(ANCHOR) != 1:
        print(f'FAIL {method}: anchor count {text.count(ANCHOR)}')
        continue
    block = TEMPLATE.format(par=par, gf=gf, fps=fps, ms=ms, note=note)
    new = text.replace(ANCHOR, block + ANCHOR)
    with io.open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(new)
    print(f'ok {method}: +{len(new) - len(text)} chars')
