#!/usr/bin/env bash
# Octo LoRA(Q/V) 微调 —— 本项目采用的 baseline，对应 checkpoint step 12000。
#
#   bash run_finetune.sh
#   WORK=/data/toy SAVE_DIR=$WORK/octo_lora_fix bash run_finetune.sh
#
# 前置：
#   1. 先用 data/lerobot_to_octo_frames.py + data/build_tfds.py 备好 TFDS 数据集
#   2. --pretrained_path 指向 Octo 官方 checkpoint 目录
set -euo pipefail

WORK="${WORK:-/root/autodl-tmp}"
PY="${PY:-/root/autodl-tmp/conda_envs/octo/bin/python}"   # octo 环境（jax / optax）
DATA_DIR="${DATA_DIR:-$WORK/octo_tfds_v2}"
SAVE_DIR="${SAVE_DIR:-$WORK/octo_lora_fix}"

export HF_HUB_CACHE="${HF_HUB_CACHE:-$WORK/hf_hub}"
mkdir -p "$SAVE_DIR"

exec "$PY" "$(dirname "$0")/finetune_octo_lora_qv.py" \
  --pretrained_path "$WORK/octo_weights" \
  --data_dir  "$DATA_DIR" \
  --save_dir  "$SAVE_DIR" \
  --batch_size 8 \
  --max_steps 16000 \
  --eval_freq 1000 \
  --lr 1e-4 \
  --lora_rank 8
