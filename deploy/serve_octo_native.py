#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""云端 **Octo-native** LoRA 真机推理服务(NATIVE 格式,非 doomed dim6/50)。

架构(与 finetune_so101_lora_native.py / verify_octo_native.py 完全对齐):
  - 纯视觉(无 proprio),window=2(模型条件在相邻 2 时刻),primary 256 / wrist 128。
  - DiffusionActionHead dim=7 horizon=4;仅 LoRA 微调 transformer,base head 保留。
  - sample_actions 返回 (1, horizon=4, dim=7) = (batch,horizon,dim),chunk = raw[0]=(4,7)。
  - 反归一: 用 model.dataset_statistics['action'](6 维 M100 mean/std),把 7 维预测反
    归一到 M100(7 维 = 6 个 M100 关节 + 0/1 的 gripper pad),取 [:, :6] 给机器人。

服务端保留 2 帧滚动窗口(prev + cur),驱动端只需每轮 POST 当前 1 帧;换任务/新回合
传 reset:true 先清 buffer。图片经 base64(PIL 解码成 RGB,无 cv2)。

与 old serve_infer_octo.py 同用 http.server(标准库,octo env 无 fastapi 也能跑)。
POST /infer -> {"action": [[...],[...]] (4,6) M100, "n_steps":4, "horizon":4, "infer_ms":..}
"""
import argparse
import base64
import io
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_CACHE", "/root/autodl-tmp/hf_hub")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.30")  # 与 3 个 SmolVLA serve + YOLO 共存留显存

import jax
import jax.numpy as jnp
from octo.model.octo_model import OctoModel

DEFAULT_TASK = "Pick up the toy and place it into the box"
HORIZON = 4
G = {"lock": threading.Lock(),
     "prev_side": None, "prev_wrist": None}

CKPT = None
MODEL = None


def load_state(ckpt, step):
    model = OctoModel.load_pretrained(ckpt, step=step)
    n = sum(x.size for x in jax.tree_util.tree_leaves(model.params))
    print(f"[load] OK: step={step} text_processor={type(model.text_processor).__name__} params={n}", flush=True)
    return model


def _denorm_stats():
    """7 维反归一统计: 6 维 M100 mean/std + append gripper (mean0, std1)."""
    astat = MODEL.dataset_statistics["action"]            # 6 维 M100
    m = np.asarray(astat["mean"], np.float32)
    s = np.asarray(astat["std"], np.float32)
    amean = jnp.concatenate([m, jnp.zeros(1, jnp.float32)])
    astd = jnp.concatenate([s, jnp.ones(1, jnp.float32)])
    return amean, astd


def decode_bgr(b64):
    """base64 -> PIL RGB -> uint8 (H,W,3). 无 cv2."""
    buf = base64.b64decode(b64)
    img = Image.open(io.BytesIO(buf)).convert("RGB")
    # NOTE(2026-09-10): 这里原本有 [:, :, ::-1], 是错的, 已删除。
    # JPEG 内部一律 YCbCr; 驱动侧 cv2.imencode 收到 BGR 数组时已做 BGR->YCbCr,
    # PIL 解出来就是正确的 RGB; 再翻一次反而变成 BGR。
    # 训练侧(TFDS)是真 RGB => 修复前每条推理都在红蓝对调下跑。
    img = np.asarray(img)
    return np.ascontiguousarray(img, dtype=np.uint8)


def prep_image(img, target):
    """RGB uint8 -> resize 到 (target,target), 用 LANCZOS(高质, ViT 友好)."""
    if (img.shape[0], img.shape[1]) != (target, target):
        pil = Image.fromarray(img).resize((target, target), Image.LANCZOS)
        img = np.asarray(pil)
    return np.ascontiguousarray(np.clip(img, 0, 255).astype(np.uint8))


def infer(side_b64, eih_b64, task, reset):
    side = prep_image(decode_bgr(side_b64), 256)
    wrist = prep_image(decode_bgr(eih_b64), 128)

    # --- 滚动 2 帧窗口 (prev, cur) ---
    if reset or G["prev_side"] is None:
        G["prev_side"], G["prev_wrist"] = side, wrist
        prev_s, prev_w = side, wrist
    else:
        prev_s, prev_w = G["prev_side"], G["prev_wrist"]
    G["prev_side"], G["prev_wrist"] = side, wrist

    observations = {
        "image_primary": jnp.asarray(np.stack([prev_s, side])[None], jnp.uint8),   # (1,2,256,256,3)
        "image_wrist":   jnp.asarray(np.stack([prev_w, wrist])[None], jnp.uint8),  # (1,2,128,128,3)
        "timestep_pad_mask": jnp.ones((1, 2), dtype=bool),
    }
    tasks = {"language_instruction": MODEL.text_processor.encode([task or DEFAULT_TASK])}

    raw = MODEL.sample_actions(observations, tasks, argmax=False, rng=jax.random.PRNGKey(42))
    norm = np.asarray(raw[0])                       # (4,7) normalized
    amean, astd = _denorm_stats()
    pred = norm * astd + amean                       # (4,7) -> M100 (7 维)
    action = np.asarray(pred)[:, :6]                 # (4,6) 真实 6 关节 M100
    action = np.clip(action, -100.0, 100.0)          # M100 关节域 [-100,100], 防扩散越界
    return action


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, {"ok": True, "state": "ready", "ckpt": CKPT, "backend": "octo-native"})

    def do_POST(self):
        if self.path != "/infer":
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n))
            t0 = time.time()
            with G["lock"]:
                action = infer(data["side"], data["eye_in_hand"],
                               data.get("task"), bool(data.get("reset", False)))
            dt = (time.time() - t0) * 1000
            self._send(200, {
                "action": action.tolist(), "action_dim": int(action.shape[1]),
                "n_steps": int(action.shape[0]), "horizon": HORIZON,
                "infer_ms": round(dt, 1), "task": data.get("task") or DEFAULT_TASK,
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send(500, {"error": str(e)})

    def log_message(self, *a):  # quiet
        pass


def main():
    global CKPT, MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=6020)
    ap.add_argument("--ckpt", default="/root/autodl-tmp/octo_lora_native_out")
    ap.add_argument("--step", type=int, default=13000)
    a = ap.parse_args()
    CKPT = a.ckpt
    MODEL = load_state(a.ckpt, a.step)
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    print(f"[serve] octo-native http://0.0.0.0:{a.port} /infer  ckpt={a.ckpt} step={a.step}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
