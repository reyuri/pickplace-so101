#!/usr/bin/env bash
# 服务器侧：起 **Octo 红框版** 推理链路（两个进程）
#   1) yolo_serve.py            -> 127.0.0.1:6090  (venv_yolo)
#   2) serve_octo_redbox.py     -> 0.0.0.0:6020    (octo env)
# 红框版必须两个都在：没有 YOLO 就只能喂不带框的图，等于分布外，动作不可信。
#
# 用法（服务器上）：
#   bash start_octo_redbox.sh [step]      # step 默认 8000
#   bash start_octo_redbox.sh stop
#
# 注意 6020 是 OCTO 与 OCTO_RB 共用的端口，同一时刻只能起一个 serve。
set -u
STEP="${1:-8000}"
ROOT=/root/autodl-tmp
PY_OCTO=$ROOT/conda_envs/octo/bin/python
PY_YOLO=$ROOT/venv_yolo/bin/python
LOGD=$ROOT/logs_redbox
PORT_SERVE=6020
PORT_YOLO=6090

mkdir -p "$LOGD"

alive() { curl -s --max-time 3 "http://127.0.0.1:$1/" >/dev/null 2>&1; }

# 这台容器里没有 ss/netstat(实测 command -v 都为空)，只能按进程名找。
if [ "$STEP" = "stop" ]; then
  for name in serve_octo_redbox.py yolo_serve.py; do
    pids=$(pgrep -f "$name" 2>/dev/null | tr '\n' ' ')
    if [ -n "$pids" ]; then
      echo "kill $name: $pids"; kill $pids 2>/dev/null; sleep 1
    else
      echo "$name: 没有进程"
    fi
  done
  alive $PORT_SERVE && echo "警告: $PORT_SERVE 仍有响应" || echo "$PORT_SERVE 已停"
  exit 0
fi

# --- 1) YOLO ---
if alive $PORT_YOLO; then
  echo "[yolo] 已在跑 ($PORT_YOLO)"
else
  echo "[yolo] 启动..."
  setsid nohup "$PY_YOLO" "$ROOT/yolo_serve.py" --port=$PORT_YOLO \
      > "$LOGD/yolo_$PORT_YOLO.log" 2>&1 </dev/null &
  for i in $(seq 1 30); do alive $PORT_YOLO && break; sleep 1; done
  if alive $PORT_YOLO; then
    echo "[yolo] OK  $(tail -1 "$LOGD/yolo_$PORT_YOLO.log")"
  else
    echo "[yolo] !! 起不来，看 $LOGD/yolo_$PORT_YOLO.log"; exit 1
  fi
fi

# --- 2) Octo 红框 serve ---
if alive $PORT_SERVE; then
  echo "[serve] $PORT_SERVE 已被占用 —— 先 bash $0 stop（6020 是 OCTO/OCTO_RB 共用端口）"
  exit 1
fi
echo "[serve] 启动 octo_lora_redbox @ step=$STEP ..."
setsid nohup "$PY_OCTO" "$ROOT/serve_octo_redbox.py" \
    --ckpt "$ROOT/octo_lora_redbox" --step "$STEP" --port $PORT_SERVE \
    > "$LOGD/serve_redbox_${STEP}.log" 2>&1 </dev/null &

for i in $(seq 1 180); do
  alive $PORT_SERVE && break
  if grep -qiE "Traceback|Error" "$LOGD/serve_redbox_${STEP}.log" 2>/dev/null; then
    echo "[serve] !! 启动报错："; tail -25 "$LOGD/serve_redbox_${STEP}.log"; exit 1
  fi
  sleep 2
done

if alive $PORT_SERVE; then
  echo "[serve] OK"; grep -E "^\[(load|serve)\]" "$LOGD/serve_redbox_${STEP}.log" | tail -3
  echo
  echo "本地开隧道: bash tunnel_dual.sh start   (6020 已在转发列表里)"
  echo "真机跑:     python drive_so101_dual.py --model OCTO_RB --toy <elephant|tree|ball> --read-only"
else
  echo "[serve] !! 120s 内没起来，看 $LOGD/serve_redbox_${STEP}.log"; tail -25 "$LOGD/serve_redbox_${STEP}.log"; exit 1
fi
