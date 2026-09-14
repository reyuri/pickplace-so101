#!/usr/bin/env bash
# 服务器侧：起 **pi0.5 LoRA** 推理链路（单进程）
#   serve_infer_pi05.py -> 0.0.0.0:6030   (smolvla env)
#
# 不需要 YOLO（pi05 训练时没有红框，喂带框的图反而是分布外）。
#
# 用法（服务器上）：
#   bash start_pi05_serve.sh [step]      # step 默认 018000
#   bash start_pi05_serve.sh stop
#
# 6030 是 pi05 独占端口，和 6020/6021(Octo) 互不干扰，可同时在线。
set -u
STEP="${1:-018000}"
ROOT=/root/autodl-tmp
PY=/root/miniconda3/envs/smolvla/bin/python   # 注意: 不在 $ROOT/conda_envs 下(那里只有 octo)
LOGD=$ROOT/logs_pi05
PORT=6030
CKPT=$ROOT/outputs/pi05_lora_r16/checkpoints/$STEP/pretrained_model
DSROOT=$ROOT/lerobot_rey_toy_all

mkdir -p "$LOGD"

alive() { curl -s --max-time 3 "http://127.0.0.1:$1/" >/dev/null 2>&1; }

# 这台容器里没有 ss/netstat(实测 command -v 都为空)，只能按进程名找。
if [ "$STEP" = "stop" ]; then
  pids=$(pgrep -f "serve_infer_pi05.py" 2>/dev/null | tr '\n' ' ')
  if [ -n "$pids" ]; then
    echo "kill serve_infer_pi05.py: $pids"; kill $pids 2>/dev/null; sleep 1
  else
    echo "serve_infer_pi05.py: 没有进程"
  fi
  alive $PORT && echo "警告: $PORT 仍有响应" || echo "$PORT 已停"
  exit 0
fi

if [ ! -d "$CKPT" ]; then
  echo "!! ckpt 目录不存在: $CKPT"; ls "$ROOT/outputs/pi05_lora_r16/checkpoints/" 2>/dev/null; exit 1
fi

if alive $PORT; then
  echo "[serve] $PORT 已被占用 —— 先 bash $0 stop"
  exit 1
fi

echo "[serve] 启动 pi05 LoRA @ step=$STEP ..."
# 离线四件套：pi05 链路会去找 tokenizer，联网会卡住/失败
HF_HOME=$ROOT/.cache/huggingface \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 \
setsid nohup "$PY" "$ROOT/serve_infer_pi05.py" \
    --port=$PORT --ckpt "$CKPT" --ds-root "$DSROOT" \
    > "$LOGD/serve_pi05_${STEP}.log" 2>&1 </dev/null &

# 加载 13.5G 基座 + 预热，给足 420s
for i in $(seq 1 210); do
  alive $PORT && break
  # 注意 SystemExit("msg") 只把 msg 打到 stderr, 不会有 "SystemExit" 字样 —— 按内容抓。
  # 先滤掉 torchcodec/libav 那段**已知非致命**的加载噪声(它会打一堆 Traceback),
  # 否则会把正常启动误判成崩溃。
  if grep -vE "torchcodec|libtorchcodec|libav|FFmpeg version|_dlopen|ctypes|load_library|load_torchcodec|OSError|Traceback|File \"|raise |from e|^\s+[~^]" \
       "$LOGD/serve_pi05_${STEP}.log" 2>/dev/null |
     grep -qE "闸门.*失败|无效|没有 model\.safetensors|不是 PI05Config|Error"; then
    echo "[serve] !! 启动被拒（闸门没过）："; tail -30 "$LOGD/serve_pi05_${STEP}.log"; exit 1
  fi
  sleep 2
done

if alive $PORT; then
  echo "[serve] OK"
  grep -E "^\[(load|gate|warmup|serve)\]" "$LOGD/serve_pi05_${STEP}.log"
  echo
  echo "本地开隧道: bash tunnel_dual.sh start   (6030 已在转发列表里)"
  echo "真机跑:     python drive_so101_dual.py --model PI05 --toy <elephant|tree|ball> --read-only"
else
  echo "[serve] !! 420s 内没起来，看 $LOGD/serve_pi05_${STEP}.log"; tail -30 "$LOGD/serve_pi05_${STEP}.log"; exit 1
fi
