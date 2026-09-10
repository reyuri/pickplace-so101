#!/usr/bin/env bash
# A1 = 基线：侧视原图，无红框
#
#   bash train_A1.sh                       # WORK 默认 /root/autodl-tmp
#   WORK=/data/toy bash train_A1.sh        # 换工作根目录
#   nohup bash train_A1.sh > train_A1.log 2>&1 &
#
# 目录约定（都在 WORK 下）：
#   lerobot/                    lerobot 源码 checkout（含本项目对 eval 切分的补丁）
#   smolvla_train_launcher.py   训练启动器（train/smolvla/ 下这份，拷到 WORK 或改下面路径）
#   lerobot_rey_toy_all/   训练数据集（LeRobot v3.0）
#   outputs/train_A1/        输出：checkpoints/005000/pretrained_model 是本项目选中的权重
set -euo pipefail

WORK="${WORK:-/root/autodl-tmp}"
CONDA_ENV="${CONDA_ENV:-smolvla}"
DATASET="lerobot_rey_toy_all"

source /root/miniconda3/etc/profile.d/conda.sh && conda activate "$CONDA_ENV"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="$WORK/.cache/huggingface" HF_HUB_DISABLE_XET=1
cd "$WORK/lerobot"

exec python "$WORK/smolvla_train_launcher.py" \
  --dataset.root="$WORK/$DATASET" \
  --steps=10000 \
  --batch_size=12 --num_workers=4 \
  --log_freq=100 \
  --save_freq=1000 \
  --eval_steps=1000 \
  --dataset.eval_split=0.11 \
  --output_dir="$WORK/outputs/train_A1" \
  --job_name=toy_all_A1 \
  --policy.repo_id="$DATASET" \
  --dataset.repo_id="$DATASET" \
  --policy.device=cuda \
  --wandb.enable=false
