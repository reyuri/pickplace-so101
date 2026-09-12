#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""云端 **Octo 红框版** 真机推理服务（octo_lora_redbox）。

和 serve_octo_native.py 的唯一区别：进模型之前，先用 YOLO 把**指令里那只玩具**框出来。
训练集 octo_tfds_redbox 的侧视图就是这副样子（bake_redbox_npz.py 烙的框），
指令串也是 A3 的 "Pick up the {toy} in red box and place it into the box"。
两边必须一致，否则模型看到的是分布外的图 —— 框不能漏，指令串也不能写错。

为什么不在这个进程里直接跑 YOLO：
  octo env 没有 ultralytics，YOLO 跑在 venv_yolo 的 yolo_serve.py（127.0.0.1:6090）。
  这里只用标准库 urllib 把**驱动端发来的那张图原样**转过去，避免二次编解码引入色差。

坐标口径（关键，别改错）：
  YOLO 返回的是它解出来的原图上的像素 xyxy。这里把它归一化成中心式 [cx,cy,w,h]，
  先 resize 到 256、**再**按归一化坐标画 2px 纯红框 —— 与 bake_redbox_npz.py 的
  draw() 几何完全一致（那边是在 256 的原图上算框再画）。
  反过来"先画后缩"会让 2px 线在缩放中糊掉，红框变淡，和训练分布就不一样了。

其余（2 帧滚动窗口、7 维反归一、纯视觉无 proprio、无 cv2 的 RGB 解码）全部沿用
serve_octo_native.py，未做任何改动，包括 decode_bgr 里那条"不要再翻通道"的注释。

用法：
  /root/autodl-tmp/conda_envs/octo/bin/python serve_octo_redbox.py \
      --ckpt /root/autodl-tmp/octo_lora_redbox --step 8000 --port 6020
"""
import argparse
import base64
import io
import json
import os
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_CACHE", "/root/autodl-tmp/hf_hub")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.30")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from octo.model.octo_model import OctoModel  # noqa: E402

# A3 模板，与 make_a3_dataset.py / _tmp_build_redbox.py 一字不差
DEFAULT_TASK = "Pick up the tree in red box and place it into the box"
HORIZON = 4
TOYS = ("elephant", "tree", "ball")
YOLO_URL = "http://127.0.0.1:6090/yolo"

TH = 2                                            # 与 bake_redbox_npz.py 一致
RED = np.array([255, 0, 0], dtype=np.uint8)       # RGB

G = {"lock": threading.Lock(), "prev_side": None, "prev_wrist": None}

CKPT = None
MODEL = None


def load_state(ckpt, step):
    model = OctoModel.load_pretrained(ckpt, step=step)
    n = sum(x.size for x in jax.tree_util.tree_leaves(model.params))
    print(f"[load] OK: step={step} text_processor={type(model.text_processor).__name__} params={n}",
          flush=True)
    return model


def _denorm_stats():
    """7 维反归一统计: 6 维 M100 mean/std + append gripper (mean0, std1)。"""
    astat = MODEL.dataset_statistics["action"]            # 6 维 M100
    m = np.asarray(astat["mean"], np.float32)
    s = np.asarray(astat["std"], np.float32)
    amean = jnp.concatenate([m, jnp.zeros(1, jnp.float32)])
    astd = jnp.concatenate([s, jnp.ones(1, jnp.float32)])
    return amean, astd


def decode_bgr(b64):
    """base64 -> PIL RGB -> uint8 (H,W,3)。无 cv2。

    NOTE(2026-09-10): 这里原本有 [:, :, ::-1]，是错的，已删除，别加回来。
    JPEG 内部一律 YCbCr；驱动侧 cv2.imencode 收到 BGR 数组时已做 BGR->YCbCr，
    PIL 解出来就是正确的 RGB；再翻一次反而变成 BGR。
    训练侧(TFDS)是真 RGB => 修复前每条推理都在红蓝对调下跑。
    """
    buf = base64.b64decode(b64)
    img = Image.open(io.BytesIO(buf)).convert("RGB")
    return np.ascontiguousarray(np.asarray(img), dtype=np.uint8)


def prep_image(img, target):
    """RGB uint8 -> resize 到 (target,target)，LANCZOS。"""
    if (img.shape[0], img.shape[1]) != (target, target):
        img = np.asarray(Image.fromarray(img).resize((target, target), Image.LANCZOS))
    return np.ascontiguousarray(np.clip(img, 0, 255).astype(np.uint8))


def toy_from_task(task):
    for t in TOYS:
        if t in (task or ""):
            return t
    return None


def yolo_box(side_b64, task):
    """问 YOLO 服务要目标玩具的像素框。返回 (xyxy|None, toy|None, err|None)。"""
    toy = toy_from_task(task)
    if toy is None:
        return None, None, "指令里没有 elephant/tree/ball，无法确定框哪只"
    body = json.dumps({"image_b64": side_b64, "task": task}).encode()
    req = urllib.request.Request(YOLO_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5.0) as r:
            j = json.loads(r.read())
    except Exception as e:
        return None, toy, "YOLO 服务(%s)不可用: %s" % (YOLO_URL, e)
    return j.get("box"), toy, (None if j.get("box") else "YOLO 未检到 %s (conf>=0.3)" % toy)


def draw_box(img, xyxy):
    """img: (H,W,3) uint8 RGB 原图；xyxy 像素框。在 **原图** 上不改，只返回归一化框。

    注意：真正落笔在 256 的那张图上做，见 prep_and_box()。
    """
    H, W = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    cx = ((x1 + x2) / 2.0) / W
    cy = ((y1 + y2) / 2.0) / H
    bw = (x2 - x1) / W
    bh = (y2 - y1) / H
    return [cx, cy, bw, bh]


def stamp(img256, cxcywh):
    """在 256 的图上按归一化中心式框画 2px 纯红框 —— 逐像素等价于 bake_redbox_npz.draw()。"""
    H, W = img256.shape[:2]
    cx, cy, bw, bh = cxcywh
    x1 = int(round((cx - bw / 2) * W)); y1 = int(round((cy - bh / 2) * H))
    x2 = int(round((cx + bw / 2) * W)); y2 = int(round((cy + bh / 2) * H))
    x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W - 1, x2))
    y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return img256
    img256[y1:min(H, y1 + TH), x1:x2 + 1] = RED
    img256[max(0, y2 - TH + 1):y2 + 1, x1:x2 + 1] = RED
    img256[y1:y2 + 1, x1:min(W, x1 + TH)] = RED
    img256[y1:y2 + 1, max(0, x2 - TH + 1):x2 + 1] = RED
    return img256


def infer(side_b64, eih_b64, task, reset):
    raw_side = decode_bgr(side_b64)
    wrist = prep_image(decode_bgr(eih_b64), 128)

    box, toy, err = yolo_box(side_b64, task)
    side = prep_image(raw_side, 256)
    if box is not None:
        side = stamp(side, draw_box(raw_side, box))
    else:
        print("[warn] %s —— 这一帧不带红框就进模型了，动作不可信" % err, flush=True)

    # --- 滚动 2 帧窗口 (prev, cur) ---
    if reset or G["prev_side"] is None:
        G["prev_side"], G["prev_wrist"] = side, wrist
        prev_s, prev_w = side, wrist
    else:
        prev_s, prev_w = G["prev_side"], G["prev_wrist"]
    G["prev_side"], G["prev_wrist"] = side, wrist

    observations = {
        "image_primary": jnp.asarray(np.stack([prev_s, side])[None], jnp.uint8),
        "image_wrist":   jnp.asarray(np.stack([prev_w, wrist])[None], jnp.uint8),
        "timestep_pad_mask": jnp.ones((1, 2), dtype=bool),
    }
    tasks = {"language_instruction": MODEL.text_processor.encode([task or DEFAULT_TASK])}

    raw = MODEL.sample_actions(observations, tasks, argmax=False, rng=jax.random.PRNGKey(42))
    norm = np.asarray(raw[0])                        # (4,7) normalized
    amean, astd = _denorm_stats()
    action = np.asarray(norm * astd + amean)[:, :6]  # (4,6) M100
    action = np.clip(action, -100.0, 100.0)
    return action, box, toy, err


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, {"ok": True, "state": "ready", "ckpt": CKPT,
                         "backend": "octo-redbox", "yolo": YOLO_URL})

    def do_POST(self):
        if self.path != "/infer":
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n))
            t0 = time.time()
            with G["lock"]:
                action, box, toy, err = infer(data["side"], data["eye_in_hand"],
                                              data.get("task"), bool(data.get("reset", False)))
            dt = (time.time() - t0) * 1000
            self._send(200, {
                "action": action.tolist(), "action_dim": int(action.shape[1]),
                "n_steps": int(action.shape[0]), "horizon": HORIZON,
                "infer_ms": round(dt, 1), "task": data.get("task") or DEFAULT_TASK,
                "box": box,                      # 驱动端拿去画显示框
                "box_toy": toy,
                "box_missing": box is None,
                "box_warn": err,
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send(500, {"error": str(e)})

    def log_message(self, *a):
        pass


def main():
    global CKPT, MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=6020)
    ap.add_argument("--ckpt", default="/root/autodl-tmp/octo_lora_redbox")
    ap.add_argument("--step", type=int, default=8000)
    a = ap.parse_args()
    CKPT = a.ckpt
    MODEL = load_state(a.ckpt, a.step)
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    print(f"[serve] octo-redbox http://0.0.0.0:{a.port} /infer  ckpt={a.ckpt} step={a.step} "
          f"yolo={YOLO_URL}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
