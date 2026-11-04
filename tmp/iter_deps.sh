#!/bin/bash
VENV=/share_datasets/Seal_Model_Reproduction/Ours/venv_ours
PY=$VENV/bin/python
cd /share_datasets/Seal_Model_Reproduction/Ours/Fully_Supervised_Code_1104
for i in 1 2 3 4 5 6; do
  out=$(SEAL_CUDA=1 $PY smoke_lora.py 2>&1)
  rc=$?
  if [ $rc -eq 0 ]; then echo "$out" | tail -25; exit 0; fi
  mod=$(echo "$out" | grep -oE "No module named '[^']+'" | head -1 | sed "s/No module named '//;s/'//")
  if [ -z "$mod" ]; then echo "---- non-import failure ----"; echo "$out" | tail -20; exit 1; fi
  pkg=${mod%%.*}
  case "$pkg" in
    cv2) pkg=opencv-python-headless ;;
    PIL) pkg=pillow ;;
    sklearn) pkg=scikit-learn ;;
    yaml) pkg=pyyaml ;;
  esac
  echo "[$i] installing for missing module $mod -> $pkg"
  $PY -m pip install --quiet "$pkg" 2>&1 | tail -2
done
echo "too many iterations"
exit 1
