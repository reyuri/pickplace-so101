#!/usr/bin/env bash
# 一条命令起全云端推理栈：
#   yolo_serve(:6090, venv_yolo) + A1(:6010) / A2(:6011) / A3(:6012) 三个 SmolVLA 服务
#
#   bash serve_dual.sh start|stop|status|restart
#
# A2/A3 会对 side 帧做红框注入（--yolo-url 指到 :6090）；A1 拿不到 --yolo-url，
# 侧视原图直接进模型 —— 这正是 A1 与 A2/A3 的唯一推理侧差异。
#
# 目录约定（都在 WORK 下）：outputs/train_A{1,2,3}/checkpoints/005000/pretrained_model
# 三个服务共享一张 4090，A1/A2/A3 同时常驻（各 ~1.5GB 显存）。
set -u

WORK="${WORK:-/root/autodl-tmp}"
VY="${VY:-$WORK/venv_yolo/bin/python}"                 # YOLO 环境（ultralytics）
PY="${PY:-/root/miniconda3/envs/smolvla/bin/python}"   # SmolVLA 环境（lerobot）
SR="$WORK/serve_smolvla_dual.py"
YS="$WORK/yolo_serve.py"
SMODELS="A1:6010 A2:6011 A3:6012"
YOLO_URL=http://127.0.0.1:6090

declare -A CKPT DS BOX
CKPT[A1]=$WORK/outputs/train_A1/checkpoints/005000/pretrained_model
CKPT[A2]=$WORK/outputs/train_A2/checkpoints/005000/pretrained_model
CKPT[A3]=$WORK/outputs/train_A3/checkpoints/005000/pretrained_model
DS[A1]=$WORK/lerobot_rey_toy_all
DS[A2]=$WORK/lerobot_rey_toy_all_redbox
DS[A3]=$WORK/lerobot_rey_toy_all_redbox_a3
BOX[A1]="" BOX[A2]=$YOLO_URL BOX[A3]=$YOLO_URL   # A1 无框, A2/A3 注入

start_smolvla() {
  local name=$1 port=$2
  if ss -ltn 2>/dev/null | grep -q ":$port "; then
    echo "kill old on :$port"; fuser -k "${port}/tcp" 2>/dev/null || true; sleep 2
  fi
  nohup $PY $SR --port="$port" --ckpt="${CKPT[$name]}" --ds-root="${DS[$name]}" --yolo-url="${BOX[$name]}" \
      > "$WORK/serve_dual_${name}.log" 2>&1 &
  echo "STARTED ${name} pid=$! port=$port yolo='${BOX[$name]}'"
}

case "${1:-start}" in
  start|restart)
    export HF_HOME=$WORK/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    if ss -ltn 2>/dev/null | grep -q ":6090 "; then
      echo "kill old yolo :6090"; fuser -k "6090/tcp" 2>/dev/null || true; sleep 2
    fi
    nohup $VY $YS --port=6090 > "$WORK/yolo_serve.log" 2>&1 &
    echo "STARTED yolo pid=$! port=6090"
    sleep 5
    for m in $SMODELS; do
      name=${m%%:*}; port=${m##*:}
      start_smolvla "$name" "$port"
    done
    ;;
  stop)
    if ss -ltn 2>/dev/null | grep -q ":6090 "; then echo "kill yolo :6090"; fuser -k "6090/tcp" 2>/dev/null || true; fi
    for m in $SMODELS; do
      name=${m%%:*}; port=${m##*:}
      if ss -ltn 2>/dev/null | grep -q ":$port "; then echo "kill :$port"; fuser -k "${port}/tcp" 2>/dev/null || true; fi
      echo "stopped $name"
    done
    ;;
  status)
    for m in $SMODELS; do
      name=${m%%:*}; port=${m##*:}
      if ss -ltn 2>/dev/null | grep -q ":$port "; then echo "$name: LISTENING :$port"; else echo "$name: DOWN :$port"; fi
    done
    if ss -ltn 2>/dev/null | grep -q ":6090 "; then echo "yolo: LISTENING :6090"; else echo "yolo: DOWN :6090"; fi
    pgrep -af 'serve_smolvla_dual|yolo_serve' || echo "(no proc)"
    ;;
  *) echo "usage: $0 start|stop|status|restart"; exit 2 ;;
esac
