# -*- coding: utf-8 -*-
"""交叉实验：真机帧上模型不读红框，是**侧视图**的锅还是**腕部图**的锅？

真机帧 Δbox=0.021（训练帧 0.252），即模型在真机输入上基本无视红框。
侧视图和腕部图是同时换掉的，得把两者拆开：

    side ∈ {train, real} × wrist ∈ {train, real}

四格各测一次「框在 tree vs 框在 ball」的动作差。哪一格 Δ 掉下来，就是哪个相机。

用法
----
  P=/root/autodl-tmp/conda_envs/octo/bin/python
  $P probe_cross.py 8000
"""
from __future__ import annotations

import json
import os
import sys

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

SEEDS = (0, 1, 2, 3)
TH = 2
RED = np.array([255, 0, 0], dtype=np.uint8)
TASK = "Pick up the tree in red box and place it into the box"
# 真机帧上 YOLO 给的像素框（640x480），探针已经量过
REAL_TREE_PX = (351, 151, 513, 342)
REAL_BALL_PX = (301, 319, 414, 446)


def to_size(rgb, n):
    if rgb.shape[:2] != (n, n):
        rgb = np.asarray(Image.fromarray(rgb).resize((n, n), Image.LANCZOS))
    return np.ascontiguousarray(rgb.astype(np.uint8))


def stamp(img, cxcywh):
    H, W = img.shape[:2]
    cx, cy, bw, bh = cxcywh
    x1 = max(0, min(W - 1, int(round((cx - bw / 2) * W))))
    x2 = max(0, min(W - 1, int(round((cx + bw / 2) * W))))
    y1 = max(0, min(H - 1, int(round((cy - bh / 2) * H))))
    y2 = max(0, min(H - 1, int(round((cy + bh / 2) * H))))
    if x2 <= x1 or y2 <= y1:
        return img
    img[y1:y1 + TH, x1:x2 + 1] = RED
    img[max(0, y2 - TH + 1):y2 + 1, x1:x2 + 1] = RED
    img[y1:y2 + 1, x1:x1 + TH] = RED
    img[y1:y2 + 1, max(0, x2 - TH + 1):x2 + 1] = RED
    return img


def px_to_norm(px, W, H):
    x1, y1, x2, y2 = px
    return [((x1 + x2) / 2) / W, ((y1 + y2) / 2) / H, (x2 - x1) / W, (y2 - y1) / H]


def main():
    step = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    model = OctoModel.load_pretrained("/root/autodl-tmp/octo_lora_redbox", step=step)

    # 训练帧
    d = np.load("/root/autodl-tmp/octo_frames/tree/ep_0.npz")
    tr_side_raw = np.asarray(d["side"][69])
    tr_wrist_small = to_size(np.asarray(d["wrist"][69]), 128)
    tb = json.load(open("/root/autodl-tmp/_probe_boxes.json"))["tree/ep_0.npz"]["boxes"]

    # 真机帧
    r_side_raw = np.asarray(Image.open("/root/autodl-tmp/dual_chunk_side.jpg").convert("RGB"))
    r_wrist_raw = np.asarray(Image.open("/root/autodl-tmp/dual_chunk_eye.jpg").convert("RGB"))
    r_wrist_small = to_size(r_wrist_raw, 128)

    variants = {
        "train侧": (tr_side_raw, tb["tree"], tb["ball"]),
        "real侧": (r_side_raw,
                   px_to_norm(REAL_TREE_PX, r_side_raw.shape[1], r_side_raw.shape[0]),
                   px_to_norm(REAL_BALL_PX, r_side_raw.shape[1], r_side_raw.shape[0])),
    }
    wrists = {"train腕": tr_wrist_small, "real腕": r_wrist_small}

    task = {"language_instruction": model.text_processor.encode([TASK])}
    A = {}
    for vn, (side_raw, b_tree, b_ball) in variants.items():
        for wn, wsmall in wrists.items():
            key = "%s+%s" % (vn, wn)
            for bn, box in (("T", b_tree), ("B", b_ball)):
                img = stamp(to_size(side_raw, 256).copy(), box)
                obs = {"image_primary": jnp.asarray(np.stack([img, img])[None], jnp.uint8),
                       "image_wrist": jnp.asarray(np.stack([wsmall, wsmall])[None], jnp.uint8),
                       "timestep_pad_mask": jnp.ones((1, 2), dtype=bool)}
                for sd in SEEDS:
                    a = model.sample_actions(obs, task, argmax=False,
                                             rng=jax.random.PRNGKey(1000 + sd))
                    A[(key, bn, sd)] = np.asarray(a[0]).astype(np.float64)
            print("  跑完 %s" % key, flush=True)

    print()
    print("=" * 74)
    print("step=%d  指令固定=%r  框 tree(参考) vs 框 ball" % (step, TASK))
    print("-" * 74)
    print("  %-14s %12s %10s %12s" % ("侧+腕", "Δ(相对)", "相干比", "换seed地板"))
    for vn in variants:
        for wn in wrists:
            key = "%s+%s" % (vn, wn)
            dAB = np.stack([A[(key, "B", s)] - A[(key, "T", s)] for s in SEEDS])
            m = dAB.mean(0)
            coh = float((m * m).sum()) / max(float(np.mean((dAB * dAB).sum(1))), 1e-12)
            mag = float(np.linalg.norm(m) / (np.linalg.norm(A[(key, "T", SEEDS[0])]) + 1e-12))
            nz = float(np.mean([np.linalg.norm(A[(key, "T", b)] - A[(key, "T", a)])
                                / (np.linalg.norm(A[(key, "T", a)]) + 1e-12)
                                for i, a in enumerate(SEEDS) for b in SEEDS[i + 1:]]))
            print("  %-14s %12.4f %10.3f %12.4f" % (key, mag, coh, nz))
    print("-" * 74)
    print("  判读：Δ 明显高于换seed地板、且相干比明显高于其噪声期望 1.75 => 这一格模型读框；")
    print("        相干比完全相干时=7.0(见 val_corrected.py:_paired 的量纲推导)，不是 [0,1]。")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
