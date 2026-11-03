#!/bin/bash
set -u
cd /share_datasets/Seal_Model_Reproduction/tmp_repos
for r in TinaLRJ/DHiF haofan72/DURD GR-ray/MIST zhanglw882/ADSUNet; do
  out=$(timeout 45 git ls-remote https://github.com/$r.git HEAD 2>&1 | head -1)
  echo "$r -> $out"
done
echo "== try zip download for DHiF =="
for br in main master; do
  curl -sSL -m 120 -o dhif.zip "https://codeload.github.com/TinaLRJ/DHiF/zip/refs/heads/$br" && sz=$(stat -c%s dhif.zip) && echo "$br size=$sz" && [ "$sz" -gt 100000 ] && break
done
file dhif.zip | head -2
