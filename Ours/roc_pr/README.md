# Ours/roc_pr：Seal test 的 ROC / P-R 曲线

像素级 ROC 与 Precision-Recall 曲线，方法集 = Ours（`Ours/test_maps`）+ 12 个全监督 IRSTD 基线（服务器 `results/<Method>/Seal/fullsup_test_maps`）。口径、指标表与读图注意事项见 [docs/Seal_ROC_PR曲线.md](../../docs/Seal_ROC_PR曲线.md)。

## 版本

一个版本一个子目录，每次重画都新建版本目录，图同时存 PNG（600 dpi）与 PDF。

| 版本 | 内容 | 备注 |
|---|---|---|
| `v1_20261105/` | Ours + 12 基线；`roc`、`pr`、`roc_pr` 三张图 + `curves.json` + `summary.md` | 首个版本；Ours 用 `Ours/test_maps`，x 轴 FPR 0–5×10⁻⁴ |

## 每个版本目录里的文件

| 文件 | 说明 |
|---|---|
| `roc.png` / `roc.pdf` | 单幅 ROC（TPR vs FPR，x 轴 ×10⁻⁴） |
| `pr.png` / `pr.pdf` | 单幅 Precision-Recall |
| `roc_pr.png` / `roc_pr.pdf` | 并排两联图：(a) ROC + (b) P-R，右侧统一图例 |
| `curves.json` | 13 个方法的逐阈值 FPR/TPR、PR 点，以及 pAUC / AP / Pd@128 / Fa@128 |
| `summary.md` | 上述指标表 |

## 重新生成

```bash
# 服务器：12 个基线的直方图（~3 KB/方法，写在 results/roc_hist/）
ssh yqwang@172.18.232.202 'cd /share_datasets/Seal_Model_Reproduction && bash tools/roc_hist_all.sh'
scp -r yqwang@172.18.232.202:/share_datasets/Seal_Model_Reproduction/results/roc_hist tmp/roc/server

# 本地：Ours 直方图 + 出图
python tools/roc_hist.py --maps Ours/test_maps --gt Ours/Seal_test_GT \
    --out tmp/roc/Ours__test_maps.json --label Ours
python tools/plot_roc_pr.py --manifest tmp/roc/manifest_v1.json \
    --outdir Ours/roc_pr/v2_YYYYMMDD --title "Seal test set (1,983 images, thermal, pixel-level)"
```

换版本时把 `--outdir` 换成新的 `v2_*` 目录即可，旧版本保持不动。
