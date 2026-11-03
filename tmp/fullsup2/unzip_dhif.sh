#!/bin/bash
set -u
cd /share_datasets/Seal_Model_Reproduction/tmp_repos
rm -rf TinaLRJ_DHiF && mkdir -p TinaLRJ_DHiF && cd TinaLRJ_DHiF && unzip -q -o ../dhif.zip && ls
D=$(ls -d */ | head -1)
echo "== inner: $D"
find "$D" -maxdepth 2 -not -path '*/.git*' | head -30
echo "== README head =="
head -40 "$D/README.md" 2>/dev/null
