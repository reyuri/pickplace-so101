# -*- coding: utf-8 -*-
"""决定性测量：模型在**真机那一帧**上到底用不用红框？

背景
----
真机跑 OCTO_RB：YOLO 框确实落在树的正中，但机械臂照旧往左边跑，不碰树。
框和指令都没问题 => 怀疑模型在真机画面上根本没读那个框。

这个脚本把"框"当成唯一变量，用**驱动端真机拍的那张图**直接测：
  指令固定 "Pick up the tree in red box..."，把红框分别画在 tree / ball / elephant / 不画，
  每个条件跑多个 seed，量动作差多少（同 seed 配对，扩散噪声共模抵消）。

判读
----
  真机帧上 Δbox ≈ 0（各 seed 一致地为零）  => 模型在这一帧上根本不用框，是画面分布问题
  真机帧上 Δbox 明显 > 0                     => 模型是读框的，故障在闭环/控制那一侧

同时跑 --train 模式：拿训练帧（模型见过的分布）做同一测量当阳性对照。
两边一比就知道是"图不像"还是"别的环节"。

用法
----
  P=/root/autodl-tmp/conda_envs/octo/bin/python
  $P probe_realframe.py real  8000        # 真机帧（已传到 /root/autodl-tmp/_real_*.jpg）
  $P probe_realframe.py train 8000        # 训练帧阳性对照
"""
from __future__ import annotations

import io
import json
import os
import sys
import urllib.request

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_CACHE", "/root/autodl-tmp/hf_hub")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.28")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from octo.model.octo_model import OctoModel  # noqa: E402

TOYS = ("tree", "ball", "elephant")
SEEDS = (0, 1, 2, 3)
TH = 2
RED = np.array([255, 0, 0], dtype=np.uint8)
YOLO_URL = "http://127.0.0.1:6090/yolo"
TASK = "Pick up the %s in red box and place it into the box"


def to_size(rgb, n):
    if rgb.shape[:2] != (n, n):
        rgb = np.asarray(Image.fromarray(rgb).resize((n, n), Image.LANCZOS))
    return np.ascontiguousarray(rgb.astype(np.uint8))


def resize256(rgb):
    return to_size(rgb, 256)


def stamp(img256, cxcywh):
    """与 bake_redbox_npz.draw() 逐像素等价的 2px 纯红框。"""
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


def yolo_norm(side_rgb, toy):
    """把原图交给 YOLO 服务（与部署同一条路），返回归一化中心式框或 None。"""
    buf = io.BytesIO()
    Image.fromarray(side_rgb).save(buf, format="JPEG", quality=90)
    import base64
    body = json.dumps({"image_b64": base64.b64encode(buf.getvalue()).decode(),
                       "task": TASK % toy}).encode()
    req = urllib.request.Request(YOLO_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        j = json.loads(r.read())
    b = j.get("box")
    if not b:
        return None
    H, W = side_rgb.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in b]
    return [((x1 + x2) / 2) / W, ((y1 + y2) / 2) / H, (x2 - x1) / W, (y2 - y1) / H], b


def rel(u, v):
    return float(np.linalg.norm(u - v) / (np.linalg.norm(v) + 1e-12))


def load_real():
    side = np.asarray(Image.open("/root/autodl-tmp/dual_chunk_side.jpg").convert("RGB"))
    wrist = np.asarray(Image.open("/root/autodl-tmp/dual_chunk_eye.jpg").convert("RGB"))
    return side, wrist


def load_train(toy="tree", f=69):
    d = np.load("/root/autodl-tmp/octo_frames/%s/ep_0.npz" % toy)
    side = np.asarray(d["side"][f])[..., :3]
    wrist = np.asarray(d["wrist"][f])[..., :3]
    spec = json.load(open("/root/autodl-tmp/_probe_boxes.json"))["%s/ep_0.npz" % toy]
    return side, wrist, spec["boxes"]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "real"
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    model = OctoModel.load_pretrained("/root/autodl-tmp/octo_lora_redbox", step=step)

    train_boxes = None
    if mode == "real":
        side_raw, wrist_raw = load_real()
        boxes = {}
        for t in TOYS:
            got = yolo_norm(side_raw, t)
            boxes[t] = got[0] if got else None
            print("YOLO 指令=%-9s -> %s" % (t, got[1] if got else "未检出"), flush=True)
    else:
        side_raw, wrist_raw, train_boxes = load_train()
        boxes = {t: train_boxes.get(t) for t in TOYS}

    side256 = resize256(side_raw)
    # 腕部相机在训练/部署里都是 128（红框只烙在侧视的 256 上）
    wrist = jnp.asarray(np.stack([to_size(wrist_raw, 128)] * 2)[None], jnp.uint8)

    conds = {}
    for t in TOYS:
        if boxes.get(t) is None:
            continue
        img = stamp(side256.copy(), boxes[t])
        conds["box=" + t] = jnp.asarray(np.stack([img, img])[None], jnp.uint8)
    conds["无框"] = jnp.asarray(np.stack([side256, side256])[None], jnp.uint8)

    task = {"language_instruction": model.text_processor.encode([TASK % "tree"])}
    A = {}
    for name, obs in conds.items():
        o = {"image_primary": obs, "image_wrist": wrist,
             "timestep_pad_mask": jnp.ones((1, 2), dtype=bool)}
        for sd in SEEDS:
            a = model.sample_actions(o, task, argmax=False, rng=jax.random.PRNGKey(1000 + sd))
            A[(name, sd)] = np.asarray(a[0]).astype(np.float64)
        print("  跑完 %s" % name, flush=True)

    names = list(conds.keys())
    ref = "box=tree" if "box=tree" in names else names[0]
    print()
    print("=" * 78)
    print("模式 = %s   step = %d   指令固定 = %r" % (mode, step, TASK % "tree"))
    print("-" * 78)
    print("  %-14s %10s %10s" % ("条件", "Δ(相对ref)", "相干比"))
    for name in names:
        d = np.stack([A[(name, s)] - A[(ref, s)] for s in SEEDS])
        m = d.mean(0)
        tot = float(np.mean((d * d).sum(1)))
        coh = float((m * m).sum())
        print("  %-14s %10.4f %10.3f" % (name, rel(A[(name, SEEDS[0])], A[(ref, SEEDS[0])]),
                                         coh / max(tot, 1e-12)))
    # 噪声地板：ref 自己换 seed
    nz = [rel(A[(ref, s2)], A[(ref, s1)]) for i, s1 in enumerate(SEEDS) for s2 in SEEDS[i + 1:]]
    print()
    print("  换 seed 噪声地板（同条件）= %.4f   |   相干比零线 ≈ 1/K = %.2f"
          % (float(np.mean(nz)), 1.0 / len(SEEDS)))
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
