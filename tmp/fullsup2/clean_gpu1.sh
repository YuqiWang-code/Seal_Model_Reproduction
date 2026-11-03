#!/bin/bash
set -u
echo "== my training processes =="
pgrep -af 'fct_sam_seal/bin/python train_seal.py' | head -10
echo "== kill everything except ILNet/SCTransNet (cuda:0) =="
for pid in $(pgrep -f 'fct_sam_seal/bin/python train_seal.py'); do
  cmd=$(tr '\0' ' ' < /proc/$pid/cmdline)
  case "$cmd" in
    *cuda:0*) echo "keep $pid";;
    *) echo "kill $pid : ${cmd:0:90}"; kill -9 $pid 2>/dev/null;;
  esac
done
sleep 3
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
