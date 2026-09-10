#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""云端 SmolVLA **双相机** 推理服务:加载 A1/A2/A3 任一 005000 微调模型,POST /infer 返回 50 步 action chunk。

与 serve_infer.py(单相机 front)的区别:
  - 输入双图:payload 含 "side"(侧视)与 "eye_in_hand"(手眼)两帧 + state + task。
  - 完整加载配方(经 probe_infer_keys.py 实证):import smolvla → from_pretrained → make_policy(cfg, ds_meta)
    → load_model_st(model.safetensors, strict=True) → make_pre_post_processors → predict_action_chunk。
  - 只喂 observation.images.side + observation.images.eye_in_hand + observation.state + task(已证不需要空槽键)。

红框注入(仅当给 --yolo-url):对 side 帧调用本地 yolo_serve(:6090,venv_yolo)拿目标框坐标,
按 bake_redbox_dataset.py 的画法烙红框(粗3 + toy/conf 标签),再用**带框 side 帧**喂 SmolVLA(A2/A3 训练分布)。
返回体含 box 坐标([x1,y1,x2,y2] 或 null),供本地显示"真实进入模型的那张侧视"。

用法(每模型一份,端口各自独立):
  python serve_smolvla_dual.py --port=6010 \
      --ckpt=/root/autodl-tmp/outputs/train_A1/checkpoints/005000/pretrained_model \
      --ds-root=/root/autodl-tmp/lerobot_rey_toy_all \
      --yolo-url=http://127.0.0.1:6090      # A2/A3 给; A1 不给(~纯推理,侧视原图)
"""
import argparse
import base64
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
import cv2
import requests

import lerobot.policies.smolvla  # noqa: F401  ← 关键: 注册 'smolvla', 否则 from_pretrained 不认

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from safetensors.torch import load_model as load_model_st

DEFAULT_TASK = "Pick up the toy and place it into the box"
RED = (0, 0, 255)          # BGR, 与 bake_redbox_dataset.py 一致
BOX_THICK = 3
G = {"lock": threading.Lock()}


def load_state(ckpt, ds_root):
    ds = LeRobotDataset("reyqiao/lerobot_rey_toy_all", root=ds_root)
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    print("[load] cfg type =", type(cfg).__name__, flush=True)
    policy = make_policy(cfg, ds_meta=ds.meta)
    load_model_st(policy, os.path.join(ckpt, "model.safetensors"), strict=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy.to(device).eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=ckpt)
    print(f"[load] OK: {type(policy).__name__} on {device}", flush=True)
    return policy, pre, post, device


def decode_bgr(b64):
    buf = np.frombuffer(base64.b64decode(b64), np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("无法解码图片")
    return bgr


def bgr_to_tensor(bgr):
    """BGR → RGB float [1,3,H,W] 0..1,与训练一致。"""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0


def yolo_box(side_bgr, task):
    """调 yolo_serve 拿目标框坐标;无框返回 None。失败(服务/网络)返回 None,幂等降级为原图。"""
    try:
        ok, buf = cv2.imencode(".jpg", side_bgr)
        if not ok:
            return None
        payload = {"image_b64": base64.b64encode(buf.tobytes()).decode("ascii"), "task": task or ""}
        r = requests.post(YOLO_URL.rstrip("/") + "/yolo", json=payload, timeout=10.0)
        r.raise_for_status()
        j = r.json()
        return j if j.get("box") else None
    except Exception as e:  # noqa: BLE001
        print(f"[warn] yolo_box 失败, 降级原图: {type(e).__name__}: {e}", flush=True)
        return None


def draw_red_box(bgr, box, toy, conf):
    """按 bake 画法烙红框(原地修改 bgr)。box=[x1,y1,x2,y2]。"""
    x1, y1, x2, y2 = box
    cv2.rectangle(bgr, (x1, y1), (x2, y2), RED, BOX_THICK)
    if toy:
        cv2.putText(bgr, f"{toy} {conf:.2f}", (x1, max(y1 - 6, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)


def infer(policy, pre, post, device, side_b64, eih_b64, state, task):
    side_bgr = decode_bgr(side_b64)
    eih_bgr = decode_bgr(eih_b64)

    box = None
    if YOLO_URL:
        tb = yolo_box(side_bgr, task)
        if isinstance(tb, dict) and tb.get("box"):  # yolo_serve 返回 dict: {box,conf,cls,toy}
            box = tb["box"]
            draw_red_box(side_bgr, box, tb.get("toy"), tb.get("conf"))

    img_side = bgr_to_tensor(side_bgr)
    img_eih = bgr_to_tensor(eih_bgr)
    st = torch.from_numpy(np.asarray(state, np.float32)).view(1, -1)
    obs = {
        "observation.images.side": img_side,
        "observation.images.eye_in_hand": img_eih,
        "observation.state": st,
        "task": [task or DEFAULT_TASK],
    }
    obs = pre(obs)
    obs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in obs.items()}
    with torch.no_grad():
        chunk = policy.predict_action_chunk(obs)
    chunk = post(chunk)
    return chunk[0].cpu().numpy(), box  # [50,6], box


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, {"ok": True, "state": "ready", "device": DEVICE, "ckpt": CKPT, "yolo": bool(YOLO_URL)})

    def do_POST(self):
        if self.path != "/infer":
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n))
            side_b64 = data["side"]
            eih_b64 = data["eye_in_hand"]
            state = data["state"]
            task = data.get("task")
            t0 = time.time()
            with G["lock"]:
                action, box = infer(POLICY, PRE, POST, DEVICE, side_b64, eih_b64, state, task)
            dt = (time.time() - t0) * 1000
            self._send(200, {
                "action": action.tolist(), "action_dim": int(action.shape[1]),
                "n_steps": int(action.shape[0]), "infer_ms": round(dt, 1),
                "box": box, "task": task,
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send(500, {"error": str(e)})

    def log_message(self, *a):  # quiet
        pass


CKPT = None
YOLO_URL = ""
POLICY = PRE = POST = None
DEVICE = None


def main():
    global CKPT, YOLO_URL, POLICY, PRE, POST, DEVICE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=6010)
    ap.add_argument("--ckpt", default="/root/autodl-tmp/outputs/train_A1/checkpoints/005000/pretrained_model")
    ap.add_argument("--ds-root", default="/root/autodl-tmp/lerobot_rey_toy_all")
    ap.add_argument("--yolo-url", default="", help="给则对 side 帧做红框注入(A2/A3); 空=纯推理(A1)")
    a = ap.parse_args()
    CKPT = a.ckpt
    YOLO_URL = a.yolo_url
    POLICY, PRE, POST, DEVICE = load_state(a.ckpt, a.ds_root)
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    print(f"[serve] listening on http://0.0.0.0:{a.port}  /infer  ckpt={a.ckpt}  yolo={bool(YOLO_URL)}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
