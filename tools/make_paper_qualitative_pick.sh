#!/bin/bash
# Build the 11-column PDDiff-style qualitative figure on the Seal test set.
#
# 列：Image / Mask / DNA-Net / MSHNet / RPCANet / HDNet / L2SKNet / BGM /
#     RPCANet++ / SAM3 / Ours
# 样例：docs/qualitative_rank_top200.json 里排名 1,2,5,6,8,11,13,15,19,21,29 的 11 张
#       （分数 = Ours 单图 F1 − 8 个基线单图 F1 均值，Ours 优势最大的若干张）
#
# 两种版式：
#   full —— PDDiff 原版：整幅二值图 + 红框 + 右下角放大插图
#   roi  —— 每个面板直接给红框区域（6 倍最近邻放大），小目标更易读
#
# 用法（服务器）： bash tools/make_paper_qualitative_pick.sh [full|roi|all] [输出目录]
#   输出目录默认 Ours/qualitative/latest；建议每次改版都给一个新目录（如
#   Ours/qualitative/v7_20261120），服务器保留历史版本、本地只下载最新一版。
set -u
MODE="${1:-all}"
OUT="${2:-Ours/qualitative/latest}"
cd "$(dirname "$0")/.." || exit 1
PY="${PY:-/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python}"

METHODS=(
  "DNA-Net=results/DNANet/Seal/test_binary"
  "MSHNet=results/MSHNet/Seal/test_binary"
  "RPCANet=results/RPCANet/Seal/test_binary"
  "HDNet=results/HDNet/Seal/test_binary"
  "L2SKNet=results/L2SKNet/Seal/test_binary"
  "BGM=results/BGM/Seal/test_binary"
  "RPCANet++=results/RPCANetpp/Seal/test_binary"
  "SAM3=Ours/SAM3-test_maps"
)
# 3D 行用概率响应图（二值图是 0/255 阶跃，画出来只有一根尖刺）
SURF=(
  "DNA-Net=results/DNANet/Seal/fullsup_test_maps"
  "MSHNet=results/MSHNet/Seal/fullsup_test_maps"
  "RPCANet=results/RPCANet/Seal/fullsup_test_maps"
  "HDNet=results/HDNet/Seal/fullsup_test_maps"
  "L2SKNet=results/L2SKNet/Seal/fullsup_test_maps"
  "BGM=results/BGM/Seal/fullsup_test_maps"
  "RPCANet++=results/RPCANetpp/Seal/fullsup_test_maps"
)
ALL="2656 2735 2185 3899 3896 2712 3398 2688 2120 2894 2888"
PICK6="2656 2735 2185 3899 3896 2712"
PICK5="3398 2688 2120 2894 2888"

build () {
  tag=$1; flag=$2; shift 2
  $PY tools/build_paper_qualitative.py --samples docs/qualitative_rank_top200.json \
    --ids "$@" --image T --methods "${METHODS[@]}" --ours Ours=Ours/test_maps \
    --surf-maps "${SURF[@]}" $flag --roi-min 64 --roi-margin 24 \
    --fig-width 9.6 --header-font 7.5 --crop3d 192 \
    --out "$OUT" --tag "$tag" >/dev/null || return 1
  echo "built $tag"
}

if [ "$MODE" = full ] || [ "$MODE" = all ]; then
  build seal_pick11 "" $ALL
  build seal_pick6 "" $PICK6
  build seal_pick5 "" $PICK5
fi
if [ "$MODE" = roi ] || [ "$MODE" = all ]; then
  build seal_pick11_roi "--roi-panels" $ALL
  build seal_pick6_roi "--roi-panels" $PICK6
  build seal_pick5_roi "--roi-panels" $PICK5
fi
ls -la "$OUT"/*_qualitative.png
