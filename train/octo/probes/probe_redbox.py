# -*- coding: utf-8 -*-
"""红框路线探针：模型到底跟不跟红框走？

背景
----
SmolVLA 的 A2/A3 靠"YOLO 把目标框烙进侧视图"把成功率从 33.3% 拉到 73.3%/80.0%。
Octo 红框版（octo_lora_redbox）按同样思路训练。但 VAL loss 在这个数据集上
分不出"模型是否真的在看目标"（README 里记过：A1 的 VAL 最低、真机最差），
所以不能用 VAL 选点，得直接量它跟不跟框。

本探针
------
取几帧"三个玩具同时在场"的画面，把红框依次画在 tree/ball/elephant 上，
指令也依次取三种，构成 (框位置 b, 指令 i) 的 3x3 条件网格。然后量：

  Δbox     固定指令，把框挪到别的玩具上  -> 框驱动
  Δlang    固定框，把指令换成别的玩具    -> 语言驱动
  Δdeploy  框和指令同时指向同一只(部署时的真实用法)，在两只玩具间比
  ΔV       固定条件，换一帧画面          -> 画面驱动（归一化用的基准）
  noise    固定条件，换扩散 seed         -> 噪声地板

  S_box = Δbox / ΔV      S_lang = Δlang / ΔV      S_deploy = Δdeploy / ΔV

红框路线的判据是 S_box / S_deploy 显著大于 S_lang：
说明信息是靠框传的，不是靠语言 —— 这正是该路线要的效果。

动作层一律「同帧同 seed 配对」，让扩散采样噪声共模抵消。

用法
----
  P=/root/autodl-tmp/conda_envs/octo/bin/python
  $P probe_redbox.py /root/autodl-tmp/octo_lora_redbox 8000
  $P probe_redbox.py /root/autodl-tmp/octo_lora_fix   12000   # 基线对照
前置：先跑 _redbox_boxes.py 生成 _probe_boxes.json
"""
from __future__ import annotations

import itertools
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
from octo.model.octo_model import OctoModel  # noqa: E402

FRAMES_ROOT = os.environ.get("OCTO_FRAMES", "/root/autodl-tmp/octo_frames")
BOXES = os.environ.get("PROBE_BOXES", "/root/autodl-tmp/_probe_boxes.json")
SEEDS = (0, 1, 2, 3, 4, 5)
TOYS = ("tree", "ball", "elephant")

TH = 2                                     # 与 bake_redbox_npz.py 一致
RED = np.array([255, 0, 0], dtype=np.uint8)  # RGB


def draw(img, b):
    """img: (H,W,3) uint8 RGB; b 归一化中心式框，4 元 [cx,cy,w,h] 或 6 元 [cls,cx,cy,w,h,conf]。

    与 bake_redbox_npz.py 的 draw() 逐像素等价（2px、纯红），
    差别只是这里收 _probe_boxes.py 写的 xywhn 四元组（训练侧那份是 6 元带 conf）。
    """
    H, W = img.shape[:2]
    o = 1 if len(b) >= 6 else 0
    cx, cy, bw, bh = b[o], b[o + 1], b[o + 2], b[o + 3]
    x1 = int(round((cx - bw / 2) * W)); y1 = int(round((cy - bh / 2) * H))
    x2 = int(round((cx + bw / 2) * W)); y2 = int(round((cy + bh / 2) * H))
    x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W - 1, x2))
    y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return img
    img[y1:min(H, y1 + TH), x1:x2 + 1] = RED
    img[max(0, y2 - TH + 1):y2 + 1, x1:x2 + 1] = RED
    img[y1:y2 + 1, x1:min(W, x1 + TH)] = RED
    img[y1:y2 + 1, max(0, x2 - TH + 1):x2 + 1] = RED
    return img


def to_resize(rgb, target):
    rgb = np.asarray(rgb)[..., :3].astype(np.float32)
    if (rgb.shape[0], rgb.shape[1]) != target:
        if rgb.shape[0] % target[0] == 0 and rgb.shape[1] % target[1] == 0:
            sy, sx = rgb.shape[0] // target[0], rgb.shape[1] // target[1]
            rgb = rgb.reshape(target[0], sy, target[1], sx, 3).mean(axis=(1, 3))
        else:
            yy = np.linspace(0, rgb.shape[0] - 1, target[0]).astype(int)
            xx = np.linspace(0, rgb.shape[1] - 1, target[1]).astype(int)
            rgb = rgb[yy][:, xx]
    return np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8))


def load_frames():
    spec = json.load(open(BOXES))
    frames = []
    for key, meta in sorted(spec.items()):
        toy, ep = key.split("/")
        d = np.load(os.path.join(FRAMES_ROOT, toy, ep))
        f, i0 = meta["f"], max(0, meta["f"] - 1)
        # 侧视：三种框位置各来一份
        side_variants = {}
        for bt in TOYS:
            if bt not in meta["boxes"]:
                continue
            s = d["side"].copy()
            draw(s[i0], meta["boxes"][bt])
            draw(s[f], meta["boxes"][bt])
            side_variants[bt] = jnp.asarray(
                np.stack([to_resize(s[i0], (256, 256)), to_resize(s[f], (256, 256))])[None], jnp.uint8)
        if len(side_variants) < 3:
            continue
        wrist = jnp.asarray(np.stack([to_resize(d["wrist"][i0], (128, 128)),
                                      to_resize(d["wrist"][f], (128, 128))])[None], jnp.uint8)
        frames.append(dict(key=key, side=side_variants, wrist=wrist, f=f))
    return frames


def rel(u, v):
    return float(np.linalg.norm(u - v) / (np.linalg.norm(v) + 1e-12))


def main() -> int:
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/octo_lora_redbox"
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    model = OctoModel.load_pretrained(ckpt, step=step)
    frames = load_frames()
    nf = len(frames)
    print("=" * 88)
    print("checkpoint = %s  step = %d" % (ckpt, step))
    print("帧 = %d（%s）" % (nf, ", ".join(fr["key"] for fr in frames)))
    print("=" * 88)

    A = {}   # (frame, box_toy, instr_toy, seed) -> 动作
    for fi, fr in enumerate(frames):
        obs_base = {"image_wrist": fr["wrist"],
                    "timestep_pad_mask": jnp.ones((1, 2), dtype=bool)}
        for bt in TOYS:
            obs = dict(obs_base, image_primary=fr["side"][bt])
            for it in TOYS:
                s = "Pick up the %s in red box and place it into the box" % it
                task = {"language_instruction": model.text_processor.encode([s])}
                for sd in SEEDS:
                    a = model.sample_actions(obs, task, argmax=False,
                                             rng=jax.random.PRNGKey(1000 + sd))
                    A[(fi, bt, it, sd)] = np.asarray(a[0]).astype(np.float64)
        print("  [帧 %d/%d] %s 跑完" % (fi + 1, nf, fr["key"]), flush=True)

    # ---- 按「组」收集每个 seed 的配对差分 ----
    # 一个「组」= 一次对照实验（例如：固定指令，把框从 tree 挪到 ball）。
    # 组内每个 seed 做同一次配对，得到一族差分向量 d_s。
    #
    # 注意：配对是「同一 seed 两次前向」，扩散噪声是共模的，在差分里精确抵消，
    # 所以 d_s 里没有独立采样噪声 —— d_s ≠ 0 就已经是条件的功劳。
    # 剩下要问的是：这个条件效应是**各 seed 一致的定向位移**（可预测），
    # 还是**随 seed 乱跳**（条件扰动了去噪轨迹，输出不可预测）？
    #   d_s = μ + ε_s ，μ 是条件效应，ε 是「条件×seed」交互项
    #   F = ‖mean_s d_s‖² / (mean_s‖d_s − mean_s d_s‖² / K)，H0(μ=0) 下 ≈ 1
    #   相干幅值取偏差修正估计 ‖μ‖ ≈ sqrt(max(‖mean d‖² − res/K, 0))
    # 不做修正的话，相干幅值会把交互项也算进去，虚高。
    G = {"box": [], "lang": [], "deploy": [], "V": []}
    for fi in range(nf):
        for it in TOYS:                                          # 固定指令、挪框
            for b1, b2 in itertools.combinations(TOYS, 2):
                G["box"].append(([A[(fi, b2, it, s)] - A[(fi, b1, it, s)] for s in SEEDS],
                                 A[(fi, b1, it, SEEDS[0])]))
        for bt in TOYS:                                          # 固定框、换指令
            for i1, i2 in itertools.combinations(TOYS, 2):
                G["lang"].append(([A[(fi, bt, i2, s)] - A[(fi, bt, i1, s)] for s in SEEDS],
                                  A[(fi, bt, i1, SEEDS[0])]))
        for t1, t2 in itertools.combinations(TOYS, 2):           # 框+指令同时指向该玩具
            G["deploy"].append(([A[(fi, t2, t2, s)] - A[(fi, t1, t1, s)] for s in SEEDS],
                                A[(fi, t1, t1, SEEDS[0])]))
        for gj in range(nf):                                     # 换帧
            if gj == fi:
                continue
            for bt in TOYS:
                for it in TOYS:
                    G["V"].append(([A[(gj, bt, it, s)] - A[(fi, bt, it, s)] for s in SEEDS],
                                   A[(fi, bt, it, SEEDS[0])]))

    def agg(gs):
        K = float(len(SEEDS))
        sig = res = refsq = 0.0
        drel = []
        for ds, ref in gs:
            D = np.stack(ds)                                    # (K, n)
            m = D.mean(0)
            sig += float((m * m).sum())                         # ‖mean_s d_s‖²
            res += float(np.mean(((D - m) ** 2).sum(1)))        # 交互项能量
            refsq += float((ref ** 2).sum())
            drel += [rel(x, ref) for x in ds]
        n = len(gs)
        r = np.sqrt(refsq / n)                                  # 参考动作的 RMS 幅值
        # 偏差修正：E‖mean_s d_s‖² = ‖μ‖² + res/K
        mu2 = max(sig / n - (res / n) / K, 0.0)
        return dict(n=n, drel=float(np.mean(drel)),
                    amp=float(np.sqrt(mu2) / r),                # 相干幅值（相对参考）
                    F=float(sig / max(res / K, 1e-12)))         # H0 下 ≈ 1

    S = {k: agg(v) for k, v in G.items()}

    # 噪声地板：同帧同句同框、只换 seed（共模不抵消，作参照）
    noise = [rel(A[(fi, bt, it, s2)], A[(fi, bt, it, s1)])
             for fi in range(nf) for bt in TOYS for it in TOYS
             for s1, s2 in itertools.combinations(SEEDS, 2)]
    noise = float(np.mean(noise))

    base = S["V"]
    print()
    print("=" * 88)
    print("每种扰动：Δ 的大小 + 是否经得起「各 seed 同向」这一关")
    print("-" * 88)
    print("  %-30s %5s %9s %9s %9s" %
          ("", "组数", "Δ(相对)", "相干幅值", "F(零假设=1)"))
    for k, lab in (("box", "Δbox  固定指令, 挪红框"),
                   ("lang", "Δlang 固定框, 换指令"),
                   ("deploy", "Δdeploy 框+指令同指一只"),
                   ("V", "ΔV    换一帧画面")):
        s = S[k]
        print("  %-30s %5d %9.4f %9.4f %9.1f"
              % (lab, s["n"], s["drel"], s["amp"], s["F"]))
    print("-" * 88)
    print("  相干幅值已做偏差修正（扣掉条件×seed 交互项），F≈1 即为与交互噪声无异")
    print("  动作层换 seed 噪声地板（不配对时的参照）= %.4f" % noise)
    print()
    print("  S_box    = 相干(Δbox)   / 相干(ΔV) = %.2f" % (S["box"]["amp"] / max(base["amp"], 1e-12)))
    print("  S_lang   = 相干(Δlang)  / 相干(ΔV) = %.2f" % (S["lang"]["amp"] / max(base["amp"], 1e-12)))
    print("  S_deploy = 相干(Δdeploy)/ 相干(ΔV) = %.2f" % (S["deploy"]["amp"] / max(base["amp"], 1e-12)))
    print("=" * 88)

    ok_box = S["box"]["F"] > 2.0
    if not ok_box:
        print("判定：Δbox 的相干幅值被修正到 0 —— 挪红框带来的动作变化与")
        print("      「红框×seed」交互噪声同量级，即不可预测。红框（这版）没进动作。")
    elif S["box"]["amp"] > S["lang"]["amp"] * 1.5:
        print("判定：Δbox 相干且明显强于 Δlang（%.4f vs %.4f）"
              % (S["box"]["amp"], S["lang"]["amp"]))
        print("      ⇒ 目标信息是经红框而非语言进入动作的。")
    else:
        print("判定：Δbox 相干，但未明显强于 Δlang（%.4f vs %.4f）"
              % (S["box"]["amp"], S["lang"]["amp"]))
        print("      ⇒ 红框起作用了，但还没强到能替代语言的程度。")
    print("      对照：ΔV 的 F = %.1f（阳性对照，视觉改变应当极相干）" % S["V"]["F"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
