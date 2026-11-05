#!/bin/bash
R=/share_datasets/Seal_Model_Reproduction
echo "=== 时间 ==="; date
echo "=== GPU ==="; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
echo "=== 进程 ==="; pgrep -af "BISF_res" | head -2
echo "=== BISF_res val 曲线（最近 12 次） ==="
grep "IoU:" $R/logs/ours_bisf_res_train.log | tail -12
echo "=== 最近日志 ==="; tail -1 $R/logs/ours_bisf_res_train.log
echo "=== 异常计数 ==="; grep -icE "traceback|error|out of memory|nan" $R/logs/ours_bisf_res_train.log 2>/dev/null
echo "=== 产物 ==="; ls -la $R/results/Ours/BISF_res/ 2>/dev/null
