#!/usr/bin/env bash
set -euo pipefail
cd /share_datasets/Seal_Model_Reproduction/MIR-Net-main
exec bash run_fixed_experiment.sh Seal "$@"
