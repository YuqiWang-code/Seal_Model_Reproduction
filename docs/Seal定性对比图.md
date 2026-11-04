# Seal 定性对比图（二值预测）

论文定性图用的二值小目标预测图，全部在服务器上由推理结果统一生成。

## 生成规则

| 项目 | 说明 |
|---|---|
| 输入 | 各方法 `results/<Method>/Seal/fullsup_test_maps/`（8 位概率图，1,983 张，原生 417×622） |
| 二值化 | `pixel >= 128`（即概率 > 0.5，与 `AS1000Metrics` 的阈值一致）→ 255，否则 0 |
| 输出 | `results/<Method>/Seal/test_binary/<stem>.png`（1,983 张/方法，0/255 单通道 PNG） |
| 脚本 | [tools/make_binary_maps.py](../tools/make_binary_maps.py)，一键入口 [tools/make_qualitative.sh](../tools/make_qualitative.sh) |

`Ours` 使用项目原有的 `Ours/test_maps/`，二值图在 `results/Ours/Seal/test_binary/`。

## 方法清单（12 个对比 + Ours）

| 序号 | 方法 | 结果目录 |
|---:|---|---|
| 1 | DNA-Net | `results/DNANet/Seal/test_binary/` |
| 2 | MSHNet | `results/MSHNet/Seal/test_binary/` |
| 3 | UIU-Net | `results/UIU-Net/Seal/test_binary/` |
| 4 | SCTransNet | `results/SCTransNet/Seal/test_binary/` |
| 5 | RPCANet | `results/RPCANet/Seal/test_binary/` |
| 6 | HDNet | `results/HDNet/Seal/test_binary/` |
| 7 | ILNet | `results/ILNet/Seal/test_binary/` |
| 8 | RPCANet++ | `results/RPCANetpp/Seal/test_binary/` |
| 9 | L2SKNet | `results/L2SKNet/Seal/test_binary/` |
| 10 | BGM | `results/BGM/Seal/test_binary/` |
| 11 | DRPCA-Net | `results/DRPCANet/Seal/test_binary/` |
| 12 | DURD | `results/DURD/Seal/test_binary/` |
| 13 | Ours（参考） | `results/Ours/Seal/test_binary/` |

## 拼图样例

样例按 GT 前景面积分位数 + 两个多目标帧确定性挑选（不按任何方法分数挑），清单见 [Seal定性对比样例.json](Seal定性对比样例.json)。每个样例在服务器 `qualitative/<stem>/` 下生成：

| 文件 | 内容 |
|---|---|
| `panel.png` | 13 行 × 4 列：RGB / 热红外 / GT / 预测叠加（绿=GT，蓝=预测） |
| `thermal.png`、`gt.png` | 单列裁图（放大，最近邻） |
| `<Method>.png` | 每方法的二值预测裁图（放大） |

生成命令：

```bash
cd /share_datasets/Seal_Model_Reproduction
bash tools/make_qualitative.sh          # 12 个方法 + Ours，含二值化与拼图
# 只做某几个方法时：
python tools/build_qualitative_panels.py --samples docs/Seal定性对比样例.json \
  --methods 'DNANet=/…/results/DNANet/Seal/test_binary' 'Ours=/…/results/Ours/Seal/test_binary' \
  --out /…/qualitative
```

## 注意事项

- 二值图严格由概率图阈值得到，没有再做归一化、形态学或面积筛选，与定量表的小目标指标同一口径。
- 概率图本身是逐图 min-max 后的结果（各方法 `infer_seal.py`），因此 0.5 阈值等价于各图动态范围中点；定性对比时应与同批定量指标一起看。
- 论文排版建议：样例行按目标尺度从小到大排列，列顺序固定为上表序号，Ours 放最后一行；`panel.png` 是 13 行竖排，可直接裁成多列拼版。
