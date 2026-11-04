#!/bin/bash
R=/share_datasets/Seal_Model_Reproduction
echo "=== 时间 ==="; date
echo "=== GPU ==="; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
echo "=== 我们的进程 ==="; pgrep -af "Fully_Supervised_Code_1104/train.py|train.py --epoch" | head -3
echo "=== B+BISF val 曲线 ==="; grep "IoU:" $R/logs/ours_bisf_train.log | tail -12
echo "=== 最近日志 ==="; tail -2 $R/logs/ours_bisf_train.log
echo "=== 异常 ==="; grep -icE "traceback|error|out of memory|nan" $R/logs/ours_bisf_train.log 2>/dev/null
echo "=== 产物 ==="; ls -la $R/results/Ours/BISF/ 2>/dev/null
