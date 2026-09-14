#!/usr/bin/env bash
# pi0.5 LoRA 微调（r16 / alpha32）—— 本项目采用的 pi0.5 基线，对应 checkpoint step 018000。
#
#   bash run_pi05_lora_r16.sh
#   WORK=/data/toy STEPS=20000 bash run_pi05_lora_r16.sh
#
# 前置：
#   1. --policy.pretrained_path 指向 pi0.5 基座（官方权重，需先放到本地）
#   2. --dataset.root 指向 LeRobot v3 数据集（3 玩具 / 135 条 / 双相机 640x480）
#
# ── 相对"照抄官方示例"必须改的几处（每处都是实测踩出来的）────────────────────
#   1) ❌ 不要加 --parallelism.dp_shard=2。lerobot 的 train.py 会硬拒：
#      ValueError: PEFT is not supported under sharded training yet.
#      正解是「不加任何 parallelism 参数」——缺省 dp_shard=1 时 resolve() 自动填
#      dp_replicate=world_size，走 DDP 分支。实测 dp_replicate=2 / dp_shard=1 正常。
#   2) + --policy.push_to_hub=false。默认 True，不加会 ValueError: 'repo_id' missing。
#   3) HF_HOME 必须指向缓存根（paligemma 假缓存挂在这下面），不要用 Octo 那套
#      HF_HUB_CACHE —— 指错会解析不到 tokenizer。
#   4) 数据集靠视频解码，离线环境要 export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1，
#      否则启动阶段会去联网拉 tokenizer 而卡死。
#   5) --eval_steps 别设太小。val 15 集约 4k 帧，单次纯前向约 3.3 分钟，40 次要 2.2 小时。
#
# ── 单卡 / 双卡 ────────────────────────────────────────────────────────────
#   默认按双卡 DDP、每卡 bs=4（effective batch = 8）。
#   实例只有一块卡时，去掉 torchrun 包装并把 --batch_size 改成 8，
#   **保持 effective batch = 8 不变**，否则与其它 run 的对照就多了一个变量：
#     "$PY/lerobot-train" ... --batch_size=8 ...
set -uo pipefail

WORK="${WORK:-/root/autodl-tmp}"
PY="${PY:-/root/miniconda3/envs/smolvla/bin}"      # pi0.5 用 lerobot 主环境，不是 octo 环境
DATA="${DATA:-$WORK/lerobot_rey_toy_all}"
CKPT="${CKPT:-$WORK/pi05_base}"
OUT="${OUT:-$WORK/outputs/pi05_lora_r16}"
STEPS="${STEPS:-20000}"
JOB="${JOB:-pi05_lora_r16}"

LOGD="$WORK/logs_pi05"
mkdir -p "$LOGD"
LOG="$LOGD/train_r16.log"
WATCHLOG="$LOGD/train_r16_watchdog.log"

# 只训注意力 q/v + 状态/动作投影。PaliGemma language_model 的 q/v 也在内 ——
# 即语言表征**可被 LoRA 改动**，language conditioning 失效不是"被冻住学不会"造成的。
TARGET='.*\.paligemma\..*\.language_model\..*\.self_attn\.(q_proj|v_proj)|.*\.gemma_expert\..*\.self_attn\.(q_proj|v_proj)|model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out)'

export HF_HOME="$WORK/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
: > "$WATCHLOG"

echo "=== 开跑前 ==="
date "+%Y-%m-%d %H:%M:%S"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
df -h "$WORK" | tail -1

# 显存采样（每 30s）
( i=0; while [ $i -lt 960 ]; do
    echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr "\n" " ")"
    sleep 30; i=$((i+1))
  done ) > "$LOGD/gpu_mem_r16.txt" 2>&1 &
SAMPLER=$!

# 看门狗：日志 15 分钟无更新即判卡死（单次 eval 约 3.3 分钟，留足余量）
( while true; do
    sleep 300
    if [ -f "$LOG" ]; then
      age=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
      if [ "$age" -gt 900 ]; then
        echo "[watchdog] $(date '+%F %T') 日志 $age 秒无更新，判卡死，中止训练" >> "$WATCHLOG"
        pkill -f "job_name=$JOB"
        break
      fi
    fi
  done ) &
WATCHER=$!

echo
echo "=== 正式训练开始（双卡 DDP, effective 8, $STEPS 步） ==="
"$PY/torchrun" --standalone --nproc-per-node=2 "$PY/lerobot-train" \
  --dataset.repo_id=lerobot_rey_toy_all \
  --dataset.root="$DATA" \
  --dataset.eval_split=0.11 \
  --policy.type=pi05 \
  --policy.pretrained_path="$CKPT" \
  --policy.push_to_hub=false \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=false \
  --policy.optimizer_lr=1.5e-5 \
  --peft.method_type=lora --peft.r=16 --peft.lora_alpha=32 --peft.lora_dropout=0.05 \
  --peft.target_modules="$TARGET" \
  --accelerator.mixed_precision=bf16 \
  --ema.enable=false \
  --steps="$STEPS" --batch_size=4 --num_workers=4 \
  --log_freq=100 --save_freq=2000 --eval_steps=1000 \
  --output_dir="$OUT" --job_name="$JOB" \
  --wandb.enable=false 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}

kill "$SAMPLER" "$WATCHER" 2>/dev/null

echo
echo "=== 退出码: $RC ==="
date "+%Y-%m-%d %H:%M:%S"
grep -iE "Effective batch size" "$LOG" | head -2
grep -iE "Train/eval split" "$LOG" | head -3
echo "=== eval_loss 轨迹 ==="
grep -oE "step [0-9]+: eval_loss=[0-9.]+" "$LOG" | head -30
echo "=== train loss 轨迹（每 2000 步取一条） ==="
grep -oE "step:[0-9]+ .*loss:[0-9.]+" "$LOG" | awk "NR%20==1" | head -15
echo "=== 看门狗记录 ==="
cat "$WATCHLOG" 2>/dev/null || echo "  (无，未触发)"
echo "=== 产物 ==="
ls -1 "$OUT/checkpoints" 2>/dev/null | head -15
du -sh "$OUT" 2>/dev/null
exit $RC
