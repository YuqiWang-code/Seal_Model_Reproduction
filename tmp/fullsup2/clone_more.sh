#!/bin/bash
set -u
echo "nvcc: $(which nvcc 2>/dev/null || echo none)  cuda: $(ls -d /usr/local/cuda* 2>/dev/null | head -2 | tr '\n' ' ')"
cd /share_datasets/Seal_Model_Reproduction/tmp_repos
for r in TinaLRJ_DHiF_retry:haofan72/DURD ZhihaoMa123_BGM:ZhihaoMa123/BGM GR-ray_MIST:GR-ray/MIST zhanglw882_ADSUNet:zhanglw882/ADSUNet TinaLRJ_DHiF:TinaLRJ/DHiF; do
  name=${r%%:*}; repo=${r##*:}
  [ -d "$name" ] && { echo "exists $name"; continue; }
  timeout 400 git clone --depth 1 https://github.com/$repo.git $name >/dev/null 2>&1 && echo "OK $name" || echo "FAIL $name ($repo)"
done
