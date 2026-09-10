#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A1 专用 TensorBoard sidecar:tail train_A1.log,把 train_loss / eval_loss / lr 写进 TB event 文件。

复用 milestone-1 的 tb_sidecar 思路,额外解析 lr(若 tracker 行含 lr 则写,否则跳过)。
用法(后台,服务器上):
  conda activate smolvla
  nohup python tb_sidecar_A1.py \
      --log /root/autodl-tmp/train_A1.log \
      --tbdir /root/autodl-tmp/outputs/train_A1/tb \
      --interval 2 > /root/autodl-tmp/tb_sidecar_A1.log 2>&1 &
本地看(另终端建隧道):
  ssh -i ~/.ssh/autodl_rey -N -L 6006:127.0.0.1:6006 connect.bjb1.seetacloud.com
"""
import argparse
import os
import re
import time

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

# train: `step:<n> ... loss:<x>`(tracker 行)
RE_TRAIN = re.compile(r"\bstep:(\d+)\b.*?\bloss:([\d.]+)\b")
# eval: `step <n>: eval_loss=Y`
RE_EVAL = re.compile(r"\bstep (\d+):\s*eval_loss=([\d.]+)\b")
# lr 与 train step 配对(在同一 tracker 行里):`step:<n> ... lr:<x>`
RE_LRLINE = re.compile(r"\bstep:(\d+)\b.*?\blr:([\d.eE+-]+)\b")


def parse_chunk(text: str):
    train, eval_, lrs = [], [], []
    for m in RE_TRAIN.finditer(text):
        train.append((int(m.group(1)), float(m.group(2))))
    for m in RE_EVAL.finditer(text):
        eval_.append((int(m.group(1)), float(m.group(2))))
    for m in RE_LRLINE.finditer(text):
        lrs.append((int(m.group(1)), float(m.group(2))))
    return train, eval_, lrs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="/root/autodl-tmp/train_A1.log")
    ap.add_argument("--tbdir", default="/root/autodl-tmp/outputs/train_A1/tb")
    ap.add_argument("--interval", type=float, default=2.0)
    args = ap.parse_args()

    if SummaryWriter is None:
        print("[tb_sidecar_A1] 缺少 tensorboard 包,退出")
        return 1

    os.makedirs(args.tbdir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.tbdir, purge_step=0)
    offset = 0
    printed = set()
    print(f"[tb_sidecar_A1] 监听 {args.log} -> {args.tbdir} (interval={args.interval}s)", flush=True)
    while True:
        try:
            size = os.path.getsize(args.log)
        except OSError:
            time.sleep(args.interval)
            continue
        if size < offset:
            offset = 0
        if size > offset:
            with open(args.log, "r", encoding="utf-8", errors="ignore") as f:
                f.seek(offset)
                chunk = f.read()
                offset = f.tell()
            train, eval_, lrs = parse_chunk(chunk)
            for step, v in train:
                if ("train", step) not in printed:
                    writer.add_scalar("train_loss", v, step)
                    printed.add(("train", step))
                    print(f"[tb_sidecar_A1] train step {step}: {v:.4f}", flush=True)
            for step, v in eval_:
                if ("eval", step) not in printed:
                    writer.add_scalar("eval_loss", v, step)
                    printed.add(("eval", step))
                    print(f"[tb_sidecar_A1] eval  step {step}: {v:.4f}", flush=True)
            for step, v in lrs:
                if ("lr", step) not in printed:
                    writer.add_scalar("lr", v, step)
                    printed.add(("lr", step))
                    print(f"[tb_sidecar_A1] lr step {step}: {v:.6f}", flush=True)
            writer.flush()
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[tb_sidecar_A1] 中断退出")
