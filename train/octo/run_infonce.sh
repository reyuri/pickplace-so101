#!/usr/bin/env bash
# 语言条件修复尝试：在 LoRA 基线上加对比损失（hinge margin 版）。
# 结论是失败的 —— 语言敏感度涨上去了但真机更差，见 README「Octo 消融：Base vs infoNCE」。
#
# 与 run_finetune.sh 唯一的差异就是多两个参数：--infonce_w / --infonce_margin
# 上游改动 base 的延续：本脚本产出的 checkpoint 由 serve 的那条 OCTO_INF(6021) 加载。
set -euo pipefail

WORK="${WORK:-/root/autodl-tmp}"
PY="${PY:-/root/autodl-tmp/conda_envs/octo/bin/python}"
DATA_DIR="${DATA_DIR:-$WORK/octo_tfds_v2}"
SAVE_DIR="${SAVE_DIR:-$WORK/octo_lora_infonce_m03}"

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
  --lora_rank 8 \
  --infonce_w 5.0 \
  --infonce_margin 0.3
