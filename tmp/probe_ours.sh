#!/bin/bash
R=/share_datasets/Seal_Model_Reproduction
echo "=== 时间 ==="; date
echo "=== GPU 总览 ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader
echo "=== GPU 进程 ==="
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader
echo "=== 我们的进程 ==="
pgrep -af "python -u train" | head -5

echo "=== 原始版 val 曲线（epoch: MAE） ==="
grep "Epoch:" $R/logs/ours_original_train.log | tail -14
echo "=== 全监督版 val 曲线（epoch: IoU） ==="
grep "#VAL#" $R/logs/ours_fullsup_train.log | tail -14
echo "=== 最近日志行 ==="
tail -2 $R/logs/ours_original_train.log
tail -2 $R/logs/ours_fullsup_train.log

echo "=== 异常检查（error/traceback/nan/out of memory） ==="
grep -iE "traceback|error|nan|out of memory" $R/logs/ours_original_train.log $R/logs/ours_fullsup_train.log 2>/dev/null | grep -v "total:" | head -5
echo "(以上为空表示无异常)"

echo "=== LoRA 结果 ==="
grep "结束" $R/Ours/weights/RGB+LoRA/train_lora.log $R/Ours/weights/T+LoRA/train_lora.log
grep "#VAL#" $R/Ours/weights/RGB+LoRA/train_lora.log | tail -4
grep "#VAL#" $R/Ours/weights/T+LoRA/train_lora.log | tail -4

echo "=== 产物 ==="
ls -la $R/results/Ours/Original/best3.pth $R/results/Ours/FullySup/ 2>/dev/null
echo "=== 磁盘 ==="
df -h /share_datasets | tail -1
