#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO 红框服务(运行于服务器 venv_yolo,监听 127.0.0.1:6090,仅服务器内部调用)。

POST /yolo  {image_b64, task} → {box:[x1,y1,x2,y2], conf, cls, toy} 或 {box:null}
规则与 bake_redbox_dataset.py 完全一致:
  - 目标类 = task 文本里的 toy(elephant/tree/ball),  最高置信 1 框。
  - 检测置信 conf=0.1 先兜底抓,  画框阈值 THR=0.3(conf<0.3 视为"无框", 返回 box=null → 保留原帧)。
  - 只返回 1 个目标框坐标,**不画框**(画框/烙进 side 帧由 serve_smolvla_dual 负责)。

跑法(服务器后台): nohup /root/autodl-tmp/venv_yolo/bin/python /root/autodl-tmp/yolo_serve.py --port=6090 \
      > /root/autodl-tmp/yolo_serve.log 2>&1 &
"""
import argparse
import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
from ultralytics import YOLO

BEST = "/root/autodl-tmp/yolo_runs/yolo11n_toy5-3/weights/best.pt"
TOY2CLS = {"elephant": 0, "tree": 1, "ball": 2}
CLS2TOY = {v: k for k, v in TOY2CLS.items()}
THR = 0.3
IMGSZ = [480, 640]
G = {"model": None}


def toy_from_task(task):
    for t in ("elephant", "tree", "ball"):
        if t in (task or ""):
            return t
    return None


def box_and_conf(task, img):
    toy = toy_from_task(task)
    if toy is None:
        return {"box": None}
    tgt = TOY2CLS[toy]
    res = G["model"].predict(source=img, imgsz=IMGSZ, conf=0.1, verbose=False)[0]
    if len(res.boxes) == 0:
        return {"box": None}
    xyxy = res.boxes.xyxy.cpu().numpy()
    conf = res.boxes.conf.cpu().numpy()
    cls = res.boxes.cls.cpu().int().tolist()
    idxs = [i for i, c in enumerate(cls) if c == tgt]
    if not idxs:
        return {"box": None}
    k = max(idxs, key=lambda i: conf[i])
    if conf[k] < THR:
        return {"box": None}
    x1, y1, x2, y2 = xyxy[k]
    return {"box": [int(x1), int(y1), int(x2), int(y2)],
            "conf": float(conf[k]), "cls": int(tgt), "toy": toy}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._send(200, {"ok": True, "state": "ready"})

    def do_POST(self):
        if self.path != "/yolo":
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n))
            buf = np.frombuffer(base64.b64decode(data["image_b64"]), np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            self._send(200, box_and_conf(data.get("task"), img))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send(500, {"error": str(e)})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=6090)
    a = ap.parse_args()
    G["model"] = YOLO(BEST)
    print(f"[yolo] model loaded from {BEST}", flush=True)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"[yolo] listening 127.0.0.1:{a.port} /yolo", flush=True)
    srv.serve_forever()
