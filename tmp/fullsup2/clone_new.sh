#!/bin/bash
set -u
cd /share_datasets/Seal_Model_Reproduction/tmp_repos
clone () {
  r=$1; name=$(echo $r | tr '/' '_')
  if [ -d "$name" ]; then echo "exists $name"; return; fi
  timeout 600 git clone --depth 1 https://github.com/$r.git $name >/dev/null 2>&1 && echo "OK $name $(du -sh $name | cut -f1)" || echo "FAIL $name"
}
clone xdFai/SCTransNet
clone Li-Haoqing/ILNet
clone xumingzhu989/HDNet-TGRS
clone JN-Yang/PConv-SDloss-Data
clone mengduann/NS-FPN
clone TinaLRJ/DHiF
clone fengyiwu98/RPCANet
echo "== trees =="
for d in xdFai_SCTransNet Li-Haoqing_ILNet xumingzhu989_HDNet-TGRS JN-Yang_PConv-SDloss-Data mengduann_NS-FPN TinaLRJ_DHiF fengyiwu98_RPCANet; do
  echo "##### $d"
  find $d -maxdepth 2 -not -path '*/.git*' | head -25
done
