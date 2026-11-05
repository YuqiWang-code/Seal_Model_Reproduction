# Ours/roc_pr：Seal test 的 ROC / P-R / Pd–Fa 曲线

曲线方法集 = Ours（`Ours/test_maps`）+ 12 个全监督 IRSTD 基线（服务器 `results/<Method>/Seal/fullsup_test_maps`）。口径、指标表与读图注意事项见 [docs/Seal_ROC_PR曲线.md](../../docs/Seal_ROC_PR曲线.md)。

## 版本

一个版本一个子目录，每次重画都新建版本目录，图同时存 PNG（600 dpi）与 PDF。

| 版本 | 内容 | 备注 |
|---|---|---|
| `v1_20261105/` | 像素级 | 首版：ROC 只画窗口内的点（末点断掉），PR 含 (0,1) 哨兵 |
| `v2_20261105/` | 像素级 | 整条折线交给坐标轴裁剪、PR 去尖刺、图例按 pAUC 排序（原曲线版） |
| `v3_20261105/` | 目标级 Pd–Fa | **两段直线**（0,0 → 拐点 → 平台）；`pd_fa.png` 线性轴、`pd_fa_log.png` log 轴（推荐） |
| `v4_20261105/` | 像素级 ROC | **两段直线**（`--two-segment`），示例图同款；PR 保持原曲线 |

两段直线的定义：平台 = 该方法在右边框处的取值（像素级取 TPR@FPR=5×10⁻⁴，目标级取 max Pd），拐点 = 沿横轴升序第一个达到 `0.98 × 平台` 的点。

## 每个版本目录里的文件

| 文件 | 说明 |
|---|---|
| `roc.png` / `roc.pdf` | 单幅 ROC（TPR vs FPR，x 轴 ×10⁻⁴） |
| `pr.png` / `pr.pdf` | 单幅 Precision-Recall |
| `roc_pr.png` / `roc_pr.pdf` | 并排两联：(a) ROC + (b) P-R，右侧统一图例 |
| `pd_fa.png` / `pd_fa.pdf` | 目标级 Pd–Fa（线性横轴） |
| `pd_fa_log.png` / `pd_fa_log.pdf` | 目标级 Pd–Fa（log 横轴） |
| `curves.json` / `pd_fa_curves.json` | 逐阈值曲线与指标；重画任意版本都靠它 |
| `summary.md` | 该版本的指标表 |

## 重新生成

```bash
# 服务器：像素级直方图（~3 KB/方法）与目标级曲线
ssh yqwang@172.18.232.202 'cd /share_datasets/Seal_Model_Reproduction && bash tools/roc_hist_all.sh'
ssh yqwang@172.18.232.202 'cd /share_datasets/Seal_Model_Reproduction && bash tools/target_curves_all.sh'   # 约 1.5 h
scp -r yqwang@172.18.232.202:/share_datasets/Seal_Model_Reproduction/results/roc_hist tmp/roc/server
scp -r yqwang@172.18.232.202:/share_datasets/Seal_Model_Reproduction/results/target_curves tmp/roc/target

# 本地：Ours 像素级直方图 + 出图
python tools/roc_hist.py --maps Ours/test_maps --gt Ours/Seal_test_GT \
    --out tmp/roc/Ours__test_maps.json --label Ours

# 像素级两段直线（v4 这一版）
python tools/plot_roc_pr.py --manifest tmp/roc/manifest_v2.json \
    --outdir Ours/roc_pr/v4_20261105 --title "Seal test (1,983 images)" --two-segment

# 目标级（v3 这一版，线性 + log 各出一次）
python tools/plot_target_curves.py --manifest tmp/roc/manifest_target_v1.json \
    --outdir Ours/roc_pr/v3_20261105 --title "Seal test (1,983 images)"
python tools/plot_target_curves.py --manifest tmp/roc/manifest_target_v1.json \
    --outdir Ours/roc_pr/v3_20261105 --title "Seal test (1,983 images)" --xscale log
```

换版本时把 `--outdir` 换成新的 `v5_*` 目录即可，旧版本保持不动；去掉 `--two-segment` 就是原始折线版。
