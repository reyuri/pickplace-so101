#!/usr/bin/env bash
# 本地→服务器(SSH)建立 5 条推理隧道: 本地 6010/6011/6012/6020/6021 -> 服务器同端口。
#   6010/6011/6012 = SmolVLA A1/A2/A3;  6020 = Octo(原生 OCTO 与红框版 OCTO_RB 共用);
#   6021 = Octo infonce 版 OCTO_INF(octo_lora_infonce_well), 与 6020 可同时在线。
# 依赖: ~/.ssh/config 已配 Host connect.bjb1.seetacloud.com(端口 + IdentityFile ~/.ssh/autodl_rey)。
#   ⚠️ 云端实例重建后 SSH 端口会变, 需同步改 ~/.ssh/config 里的 Port。
# 用法(本地 Git Bash): bash tunnel_dual.sh {start|stop|status}
#   start 后, 本地脚本 drive_so101_dual.py 就可访问 http://127.0.0.1:6010/6011/6012/6020/6021。
#
# Windows 可靠性: 不用 pgrep 匹配 ssh(在 Windows 匹配不到 ssh.exe 的命令行, 之前 status 一直误报 dead、
# stop 也杀不掉), 改用 `netstat -ano` 找真正监听的 PID 来做 stop/状态检测。
set -u
REMOTE=connect.bjb1.seetacloud.com
LOGDIR=/tmp
TUN_LOG=$LOGDIR/tunnel_dual.log
PORTS="6010 6011 6012 6020 6021"
NPORTS=5

# 返回监听在 127.0.0.1:$port 或 [::1]:$port 上的 PID(去重取首个); 无则空。
# netstat 列: $1=Proto $2=LocalAddr $3=ForeignAddr $4=State $5=PID。127.0.0.1:6010 与 [::1]:6010 命中同 PID。
listener_pid() {
  local port="$1"
  netstat -ano 2>/dev/null |
    awk -v p=":$port" '$4=="LISTENING" && index($2,p)>0 {print $5}' |
    sort -u | head -n1
}

# 汇总所有相关端口上正在监听的隧道 PID(去重, 空格分隔)。
tunnel_pids() {
  for p in $PORTS; do
    listener_pid "$p"
  done | sort -u
}

start() {
  stop >/dev/null 2>&1   # 先清掉占住端口的旧隧道(靠 netstat 找到并 taskkill)
  nohup ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -N \
    -L 6010:127.0.0.1:6010 -L 6011:127.0.0.1:6011 -L 6012:127.0.0.1:6012 \
    -L 6020:127.0.0.1:6020 -L 6021:127.0.0.1:6021 \
    "$REMOTE" > "$TUN_LOG" 2>&1 &
  echo "tunnel launcher pid=$!  (6010/6011/6012/6020/6021 -> server)"
  sleep 3
  status
}

stop() {
  local pids
  pids=$(tunnel_pids)
  if [ -n "$pids" ]; then
    echo "kill tunnel pids: $pids"
    for pid in $pids; do
      taskkill //F //T //PID "$pid" >/dev/null 2>&1
    done
    sleep 1
  else
    echo "no tunnel running (no listener on $PORTS)"
  fi
}

status() {
  local n=0 p
  for p in $PORTS; do
    local pid
    pid=$(listener_pid "$p")
    if curl -s --max-time 3 "http://127.0.0.1:$p/" >/dev/null 2>&1; then
      echo "$p: serve=OK  tunnel=${pid:-none}"
    else
      echo "$p: serve=NOT  tunnel=${pid:-none}"
    fi
    [ -n "$pid" ] && n=$((n+1))
  done
  if [ "$n" -eq "$NPORTS" ]; then
    echo "=> 隧道 $NPORTS 条全部在监听, 服务器 serve 就绪。"
  elif [ "$n" -gt 0 ]; then
    echo "=> 隧道部分在($n/$NPORTS)。"
  else
    echo "=> 无隧道监听(用 bash tunnel_dual.sh start 建立)。"
  fi
}

case "${1:-start}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  *) echo "usage: $0 start|stop|status"; exit 2 ;;
esac
