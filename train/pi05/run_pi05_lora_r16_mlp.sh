#!/usr/bin/env bash
# pi0.5 LoRA 微调（r16 / alpha32）—— 本项目**当前采用**的版本，对应 checkpoint step 020000。
#
#   bash run_pi05_lora_r16_mlp.sh
#   WORK=/data/toy STEPS=20000 bash run_pi05_lora_r16_mlp.sh
#
# ── 相对 run_pi05_lora_r16.sh（上一版）的改动 ────────────────────────────────
#   A) target_modules 从 74 个模块扩到 128 个：
#        上一版 = VLM attn q/v + expert attn q/v + action_in/out_proj   (3,130,368)
#        本  版 = 上一版 + expert mlp {gate,up,down}_proj               (7,554,048, +4.42 M)
#      ⇒ 这一版与上一版的**唯一自变量是「动作专家里被 LoRA 覆盖的模块范围」**，
#        其余超参（lr / rank / alpha / dropout / 步数 / 数据集 / 冻结项）逐字未动。
#   B) 删掉 regex 里三个**从未命中**的死分支：state_proj / action_time_mlp_in / action_time_mlp_out。
#      基座里根本没有这些模块名 —— state 是作为 18 个逐位数字 token 喂进 prompt 的，
#      时间 MLP 的真实名字是 time_mlp_in / time_mlp_out（**不带 action_ 前缀**）。
#      删掉行为中性，只是不让下一个读脚本的人被误导。
#   C) 单卡执行：不做 torchrun 包装，--batch_size 4 -> 8，**保持 effective batch = 8 不变**。
#      实例只有一块卡时用这个；有两块卡可自行加回 torchrun --nproc-per-node=2 并把
#      batch_size 改回 4（见上一版脚本）。
#   D) 输出 / 日志换新名字，**绝不覆盖上一版目录** —— 上一版是唯一的对照组。
#
# ── 开跑前闸门（重要）──────────────────────────────────────────────────────────
#   `--peft.target_modules` **匹配失败是静默的**：regex 写错一个字符，训练照样跑完 20000 步，
#   只是什么都没训到。所以 check_target_modules.py 会在启动前用基座 ckpt 的 header
#   独立算一遍「这个 regex 会挂哪些模块、多少参数」，要求：
#     新 regex 的模块集 == 上一版 adapter 实测的 74 个模块 + 54 个 expert MLP 模块
#     一个不多一个不少，且可训练量正好 7,554,048
#   **数不对就不开跑**（脚本非零退出，不会进训练）。
#
# 前置（同上一版）：
#   1. --policy.pretrained_path 指向 pi0.5 基座（官方权重，需先放到本地）
#   2. --dataset.root 指向 LeRobot v3 数据集（3 玩具 / 135 条 / 双相机 640x480）
#   3. ❌ 不要加 --parallelism.dp_shard=2（PEFT 与 sharded training 不兼容，train.py 会硬拒）
#   4. + --policy.push_to_hub=false（默认 True，不加会 ValueError: 'repo_id' missing）
#   5. HF_HOME 指向缓存根（paligemma 假缓存挂在这下面），离线环境要 export HF_HUB_OFFLINE=1
set -uo pipefail

WORK="${WORK:-/root/autodl-tmp}"
PY="${PY:-/root/miniconda3/envs/smolvla/bin}"      # pi0.5 用 lerobot 主环境，不是 octo 环境
DATA="${DATA:-$WORK/lerobot_rey_toy_all}"
CKPT="${CKPT:-$WORK/pi05_base}"
OUT="${OUT:-$WORK/outputs/pi05_lora_r16_mlp}"
STEPS="${STEPS:-20000}"
JOB="${JOB:-pi05_lora_r16_mlp}"
HERE="$(cd "$(dirname "$0")" && pwd)"

LOGD="$WORK/logs_pi05"
mkdir -p "$LOGD"
LOG="$LOGD/train_r16_mlp.log"
WATCHLOG="$LOGD/train_r16_mlp_watchdog.log"

# 上一版（注意力 q/v + 动作投影）+ 动作专家的 MLP gate/up/down；死分支已删。
TARGET='.*\.paligemma\..*\.language_model\..*\.self_attn\.(q_proj|v_proj)|.*\.gemma_expert\..*\.(self_attn\.(q_proj|v_proj)|mlp\.(gate_proj|up_proj|down_proj))|model\.(action_in_proj|action_out_proj)'

export HF_HOME="$WORK/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
: > "$WATCHLOG"

echo "=== 开跑前 ==="
date "+%Y-%m-%d %H:%M:%S"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
df -h "$WORK" | tail -1

# ── 闸门 0：不要覆盖上一版目录（要续训就 RESUME=1）────────────────────────────
if [ -e "$OUT" ] && [ "${RESUME:-0}" != "1" ]; then
  echo "!! $OUT 已存在 —— 预期这是全新 run。"
  echo "   要续训：RESUME=1 重跑本脚本；要重开：先 mv 走旧目录。"
  exit 2
fi

# ── 闸门 1：target_modules 到底挂了什么（静默失败的唯一防线）─────────────────
echo
echo "=== 闸门：target_modules 模块核查 ==="
if [ -f "$HERE/check_target_modules.py" ]; then
  if ! "$PY/python" "$HERE/check_target_modules.py"; then
    echo
    echo "!! 闸门 FAIL：target_modules 命中的模块集不符合预期，**不启动训练**。"
    exit 2
  fi
else
  echo "  (跳过：找不到 $HERE/check_target_modules.py，自担风险)"
fi

# 显存采样（每 30s）
( i=0; while [ $i -lt 960 ]; do
    echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr "\n" " ")"
    sleep 30; i=$((i+1))
  done ) > "$LOGD/gpu_mem_r16_mlp.txt" 2>&1 &
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
echo "=== 正式训练开始（单卡 bs=8, effective 8, $STEPS 步） ==="
"$PY/lerobot-train" \
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
  --steps="$STEPS" --batch_size=8 --num_workers=4 \
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
echo "=== 可训练参数量（应从日志里也能读到 7,554,048） ==="
grep -iE "trainable|Trainable" "$LOG" | head -5
echo "=== eval_loss 轨迹 ==="
grep -oE "step [0-9]+: eval_loss=[0-9.]+" "$LOG" | head -30
echo "=== train loss 轨迹（每 2000 步取一条） ==="
grep -oE "step:[0-9]+ .*loss:[0-9.]+" "$LOG" | awk "NR%20==1" | head -15
echo "=== 看门狗记录 ==="
cat "$WATCHLOG" 2>/dev/null || echo "  (无，未触发)"
echo "=== 产物 ==="
ls -1 "$OUT/checkpoints" 2>/dev/null | head -15
du -sh "$OUT" 2>/dev/null
echo "=== adapter 指纹（部署时用来认模型） ==="
echo "  step 020000 期望: 256 张量 / 7,554,048 参数 / 128 个模块"
echo "  （上一版 step 018000 对照: 148 张量 / 3,130,368 参数 / 74 个模块）"
exit $RC
