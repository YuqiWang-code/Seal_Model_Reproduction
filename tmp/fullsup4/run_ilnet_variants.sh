#!/bin/bash
set -e
cd /share_datasets/Seal_Model_Reproduction/ILNet-main
CUDA_VISIBLE_DEVICES=99 /home/yqwang/miniforge3/envs/fct_sam_seal/bin/python \
  /share_datasets/Seal_Model_Reproduction/tmp/fullsup4/ilnet_variants.py
