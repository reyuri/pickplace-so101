#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""红框版 Octo serve 的冒烟测试（在**服务器上**跑，因为要读 octo_frames 和本地 6090/6020）。

它替真机做两件事：
  1) 打通链路：发一张真帧 -> serve -> YOLO(6090) -> Octo -> 拿回动作，看有没有报错、耗时多少。
  2) 把服务端认定的框画出来存成 PNG，**目视确认框在哪只玩具上**。
     框错了模型就是在按错的指令抓，但输出看起来"正常"，光看动作数值发现不了。

用法:
  /root/autodl-tmp/conda_envs/octo/bin/python smoke_octo_redbox.py \
      [--host 127.0.0.1] [--port 6020] [--toy tree] [--frame 90] [--out /root/autodl-tmp/_smoke_rb.png]
"""
import argparse
import base64
import io
import json
import os
import urllib.request

import numpy as np
from PIL import Image

TOYS = ("elephant", "tree", "ball")
RED = (255, 0, 0)


def post(url, obj, timeout=120.0):
    req = urllib.request.Request(url, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def jb64(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb).convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def draw(rgb, xyxy, th=2):
    img = rgb.copy()
    H, W = img.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W - 1, x2))
    y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H - 1, y2))
    c = np.array(RED, np.uint8)
    img[y1:min(H, y1 + th), x1:x2 + 1] = c
    img[max(0, y2 - th + 1):y2 + 1, x1:x2 + 1] = c
    img[y1:y2 + 1, x1:min(W, x1 + th)] = c
    img[y1:y2 + 1, max(0, x2 - th + 1):x2 + 1] = c
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6020)
    ap.add_argument("--frames", default="/root/autodl-tmp/octo_frames")
    ap.add_argument("--toy", default="tree", choices=TOYS, help="取哪条 episode 的画面")
    ap.add_argument("--frame", type=int, default=90)
    ap.add_argument("--all-toys", action="store_true",
                    help="只用这一帧，但依次发三条指令，检查框是否跟着指令换目标")
    ap.add_argument("--out", default="/root/autodl-tmp/_smoke_rb.png")
    a = ap.parse_args()

    base = "http://%s:%d" % (a.host, a.port)
    with urllib.request.urlopen(base + "/", timeout=10) as r:
        print("健康检查 %s/ ->" % base, json.loads(r.read()))

    d = np.load(os.path.join(a.frames, a.toy, "ep_0.npz"))
    n = d["side"].shape[0]
    f = min(a.frame, n - 1)
    side = np.asarray(d["side"][f])[..., :3]
    wrist = np.asarray(d["wrist"][f])[..., :3]
    print("帧: %s/ep_0.npz  f=%d/%d  side=%s" % (a.toy, f, n, side.shape))

    todo = TOYS if a.all_toys else (a.toy,)
    tiles = []
    for toy in todo:
        task = "Pick up the %s in red box and place it into the box" % toy
        try:
            resp = post(base + "/infer", {
                "side": jb64(side), "eye_in_hand": jb64(wrist),
                "state": [0.0] * 6, "task": task, "reset": True})
        except Exception as e:
            print("  指令=%-9s !! 请求失败: %s" % (toy, e))
            continue
        act = np.asarray(resp["action"], np.float32)
        box = resp.get("box")
        print("  指令=%-9s box=%-28s toy=%-8s infer=%.0fms  动作[0]=%s"
              % (toy, box, resp.get("box_toy"), resp.get("infer_ms", -1),
                 np.round(act[0], 2).tolist()))
        if resp.get("box_warn"):
            print("              警告: %s" % resp["box_warn"])
        img = draw(side, box) if box else side
        tiles.append(np.concatenate([img, np.full((6, img.shape[1], 3), 255, np.uint8)], 0))

    if tiles:
        h = min(t.shape[0] for t in tiles)
        Image.fromarray(np.concatenate([t[:h] for t in tiles], 1)).save(a.out)
        print("\n已存框图: %s   (从左到右依次是指令 %s)" % (a.out, "/".join(todo)))
        print("请目视确认：红框落在**指令说的那只**玩具上。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
