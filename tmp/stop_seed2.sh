#!/bin/bash
# 停止 seed2 的两个训练进程（保留 seed1 全监督版与已完成的原始版）
R=/share_datasets/Seal_Model_Reproduction
echo "=== 待停止的 seed2 进程 ==="
PIDS=$(pgrep -f "FullySup_seed2|Original_seed2" | tr '\n' ' ')
echo "pids: $PIDS"

for p in $PIDS; do
  kill -TERM "$p" 2>/dev/null || true
done
sleep 15
LEFT=$(pgrep -f "FullySup_seed2|Original_seed2" | tr '\n' ' ')
if [ -n "$LEFT" ]; then
  echo "TERM 后仍在: $LEFT，改用 KILL"
  for p in $LEFT; do kill -KILL "$p" 2>/dev/null || true; done
  sleep 5
fi
echo "=== 剩余 seed2 进程 ==="
pgrep -af "FullySup_seed2|Original_seed2" || echo "(无)"

echo "=== seed1 状态 ==="
pgrep -f "results/Ours/FullySup " > /dev/null && echo "seed1 训练仍在运行" || echo "警告：seed1 未找到"
grep 'IoU:' $R/logs/ours_fullsup_train.log | tail -2
tail -1 $R/logs/ours_fullsup_train.log

echo "=== GPU ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
