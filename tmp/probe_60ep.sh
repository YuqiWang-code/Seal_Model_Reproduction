#!/bin/bash
R=/share_datasets/Seal_Model_Reproduction
echo "=== 时间 ==="; date
echo "=== GPU ==="; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
echo "=== 进程 ==="; pgrep -af "BISF_60ep" | head -2
echo "=== 60ep val 曲线（最近 12 次） ==="; grep "IoU:" $R/logs/ours_bisf_60ep.log | tail -12
echo "=== 最近日志 ==="; tail -1 $R/logs/ours_bisf_60ep.log
echo "=== 异常计数 ==="; grep -icE "traceback|error|out of memory|nan" $R/logs/ours_bisf_60ep.log 2>/dev/null
echo "=== 产物 ==="; ls -la $R/results/Ours/BISF_60ep/ 2>/dev/null
echo "=== 30ep 目录 ==="; ls $R/results/Ours/BISF/ 2>/dev/null
