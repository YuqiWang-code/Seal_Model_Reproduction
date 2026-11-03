#!/bin/bash
set -u
cd /share_datasets/Seal_Model_Reproduction/tmp_repos
for r in jzw0707/PDDiff xdFai/SCTransNet xumingzhu989/HDNet-TGRS Li-Haoqing/ILNet mengduann/NS-FPN TinaLRJ/DHiF JN-Yang/PConv-SDloss-Data fengyiwu98/RPCANet liboeryuan/PGNet ZhihaoMa123/BGM Linaom1214/ISTD-Benchmark; do
  out=$(timeout 40 git ls-remote https://github.com/$r.git HEAD 2>&1 | head -1)
  echo "$r -> $out"
done
