#!/bin/bash
set -u
cd /share_datasets/Seal_Model_Reproduction/tmp_repos
for r in TinaLRJ/DHiF fengyiwu98/RPCANet; do
  name=$(echo $r | tr '/' '_')
  [ -d "$name" ] && { echo "exists $name"; continue; }
  timeout 900 git clone --depth 1 https://github.com/$r.git $name >/dev/null 2>&1 && echo "OK $name" || echo "FAIL $name"
done
echo "== PConv tree =="
find JN-Yang_PConv-SDloss-Data -maxdepth 2 -not -path '*/.git*' -not -path '*/weight*' -not -path '*/data*' | head -30
du -sh JN-Yang_PConv-SDloss-Data/* 2>/dev/null | sort -h | tail -6
