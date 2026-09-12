# -*- coding: utf-8 -*-
"""分层诊断：语言信号在「动作头输入」和「最终动作」上各剩多少？

背景
----
`octo_lora_fix/12000` 真机上只抓离侧视相机最近的那只玩具，无视 task 串。
已知**最终动作**上的语言敏感度只有 0.156（归一化域，满量程 ±100）。
但"低"可能来自两个完全不同的地方：

  (a) ViT / T5 / transformer 就没有把语言编进 readout —— 输入侧根本没这个信息
  (b) readout 里有，是 DiffusionActionHead 在前向/去噪过程里把它冲掉了

这个区分决定下一步能不能用**输出级辅助损失**去修：

  * 若是 (a)：gradient 再怎么加也变不出数据里没有的对应关系，只能改数据（重采配对样本）
  * 若是 (b)：语言在动作头输入处还在，只是没被用上，输出级损失有戏

度量
----
对每一层 L 算无量纲比值

    S_L = ΔL_L / ΔV_L

    ΔL_L = 同一帧、换指令带来的变化（语言驱动）
    ΔV_L = 同一条指令、换帧带来的变化（视觉驱动）

两者同尺度，所以 S 可以**跨层比较**：
    S_readout >> S_action  ⇒ 语言在 readout 里，被动作头冲淡了  ⇒ 方案 (b)
    S_readout ≈ S_action    ⇒ 语言从一开始就没进来                ⇒ 方案 (a)

注：readout 是确定性的（无采样噪声），动作层要用「同 seed 换指令」配对测量，
让扩散采样的共模噪声抵消；同句异 seed 的差用来给动作层定噪声地板。

用法
----
    P=/root/autodl-tmp/conda_envs/octo/bin/python
    $P diag_lang_layers.py /root/autodl-tmp/octo_lora_fix 12000
"""
from __future__ import annotations

import glob
import itertools
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_CACHE", "/root/autodl-tmp/hf_hub")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.30")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from octo.model.octo_model import OctoModel  # noqa: E402

FRAMES_ROOT = os.environ.get("OCTO_FRAMES", "/root/autodl-tmp/octo_frames")
FRAMES_PER_TOY = int(os.environ.get("FRAMES_PER_TOY", "2"))
SEEDS = (0, 1, 2, 3)

TASKS = {
    "tree": "Pick up the tree and place it into the box",
    "ball": "Pick up the ball and place it into the box",
    "elephant": "Pick up the elephant and place it into the box",
    "empty": "",
    "gibberish": "Banana helicopter purple",
}
REF = "tree"


def to_resize(rgb, target):
    """和 probe_lang_fixeddim.py 保持一致：面积平均下采样，别用别的方式。"""
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
    """每类玩具取 FRAMES_PER_TOY 集，各取中间一帧（窗口的第二帧）。"""
    frames = []
    for toy in ("tree", "ball", "elephant"):
        paths = sorted(glob.glob(os.path.join(FRAMES_ROOT, toy, "ep_*.npz")))
        for p in paths[:FRAMES_PER_TOY]:
            d = np.load(p)
            n = d["side"].shape[0]
            f = min(n // 2, n - 1)
            i0 = max(0, f - 1)
            obs = {
                "image_primary": jnp.asarray(np.stack([to_resize(d["side"][i0], (256, 256)),
                                                       to_resize(d["side"][f], (256, 256))])[None],
                                             jnp.uint8),
                "image_wrist": jnp.asarray(np.stack([to_resize(d["wrist"][i0], (128, 128)),
                                                     to_resize(d["wrist"][f], (128, 128))])[None],
                                           jnp.uint8),
                "timestep_pad_mask": jnp.ones((1, 2), dtype=bool),
            }
            frames.append({"toy": toy, "name": os.path.basename(p), "obs": obs})
    return frames


def get_params(model):
    for attr in ("params", "_params"):
        v = getattr(model, attr, None)
        if v is not None:
            return v
    raise RuntimeError("拿不到 OctoModel 的 params（属性名变了？）")


def main() -> int:
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/octo_lora_fix"
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 12000

    print("=" * 84)
    print("checkpoint = %s  step = %d" % (ckpt, step))
    model = OctoModel.load_pretrained(ckpt, step=step)
    bm = model.module.bind({"params": get_params(model)}, rngs={"dropout": jax.random.PRNGKey(0)})

    frames = load_frames()
    print("帧 = %d（%s）  指令 = %s" % (len(frames), ", ".join(f["name"] for f in frames),
                                     "/".join(TASKS)))
    print("=" * 84)

    # ---- 逐帧、逐指令取 readout（确定性）与动作（多种子） ----
    Z = {}   # (frame_idx, instr) -> readout 向量 (W*T*D,)
    A = {}   # (frame_idx, instr, seed) -> 动作 (H, dim)
    for fi, fr in enumerate(frames):
        for k, s in TASKS.items():
            task = {"language_instruction": model.text_processor.encode([s])}
            emb = bm.octo_transformer(fr["obs"], task, fr["obs"]["timestep_pad_mask"], train=False)
            Z[(fi, k)] = np.asarray(emb["readout_action"].tokens).reshape(-1).astype(np.float64)
            for sd in SEEDS:
                a = model.sample_actions(fr["obs"], task, argmax=False,
                                         rng=jax.random.PRNGKey(1000 + sd))
                A[(fi, k, sd)] = np.asarray(a[0]).astype(np.float64)
        print("  已跑完 %s (%s)" % (fr["toy"], fr["name"]), flush=True)

    def rel(u, v):
        return float(np.linalg.norm(u - v) / (np.linalg.norm(v) + 1e-12))

    # ---- readout 层（确定性，无噪声地板） ----
    dLz, dVz = [], []
    for fi in range(len(frames)):
        for k in TASKS:
            if k != REF:
                dLz.append(rel(Z[(fi, k)], Z[(fi, REF)]))
        for gi in range(len(frames)):
            if gi != fi:
                dVz.append(rel(Z[(gi, REF)], Z[(fi, REF)]))
    dLz, dVz = float(np.mean(dLz)), float(np.mean(dVz))

    # ---- 动作层 ----
    peak = max(float(np.abs(v).max()) for v in A.values())
    dLa, dVa, noise = [], [], []
    for fi in range(len(frames)):
        for sd in SEEDS:
            for k in TASKS:
                if k != REF:
                    dLa.append(rel(A[(fi, k, sd)], A[(fi, REF, sd)]))      # 同帧同 seed 换指令
            for gi in range(len(frames)):
                if gi != fi:
                    dVa.append(rel(A[(gi, REF, sd)], A[(fi, REF, sd)]))    # 同指令同 seed 换帧
        for k in TASKS:
            for s1, s2 in itertools.combinations(SEEDS, 2):
                noise.append(rel(A[(fi, k, s2)], A[(fi, k, s1)]))          # 同帧同句换 seed
    dLa, dVa, noise = float(np.mean(dLa)), float(np.mean(dVa)), float(np.mean(noise))

    # ---- 空指令 / 胡言乱语 单独看 ----
    dLz_empty = float(np.mean([rel(Z[(fi, "empty")], Z[(fi, REF)]) for fi in range(len(frames))]))
    dLa_empty = float(np.mean([rel(A[(fi, "empty", sd)], A[(fi, REF, sd)])
                               for fi in range(len(frames)) for sd in SEEDS]))

    # ---- 输出 ----
    print()
    print("=" * 84)
    print("相对变化量（均已除以被比较对象的范数，无量纲）")
    print("-" * 84)
    print("  %-34s %10s %10s %10s" % ("", "ΔL(换指令)", "ΔV(换帧)", "S=ΔL/ΔV"))
    print("  %-34s %10.4f %10.4f %10.2f" % ("readout_action.tokens", dLz, dVz, dLz / max(dVz, 1e-12)))
    print("  %-34s %10.4f %10.4f %10.2f" % ("最终动作 (horizon×dim)", dLa, dVa, dLa / max(dVa, 1e-12)))
    print()
    print("  动作层噪声地板（同帧同句、换 seed） = %.4f" % noise)
    print("  → 动作层语言信噪比 ΔL/noise = %.2fx" % (dLa / max(noise, 1e-12)))
    print()
    print("  空指令 vs 正确指令：readout 变化 %.4f   动作变化 %.4f" % (dLz_empty, dLa_empty))
    print("  全部动作样本的峰值 |a| = %.3f（归一化域；满量程约 ±1，真机口径 ×100）" % peak)
    print("=" * 84)

    Sz = dLz / max(dVz, 1e-12)
    Sa = dLa / max(dVa, 1e-12)
    print()
    if Sz > Sa * 3:
        print("判定：语言在 readout 里明显更强（S_readout/S_action = %.1fx）" % (Sz / max(Sa, 1e-12)))
        print("      ⇒ 信息进了动作头输入，是被 head 冲掉的 ⇒ 输出级辅助损失有戏")
    elif Sa > Sz * 3:
        print("判定：动作层反而比 readout 更敏感（S_action/S_readout = %.1fx）" % (Sa / max(Sz, 1e-12)))
        print("      ⇒ 语言不是被 head 冲掉的，得往更前面找（ViT/T5/transformer）")
    else:
        print("判定：两层 S 同量级（readout %.2f vs action %.2f）" % (Sz, Sa))
        print("      ⇒ 语言在动作头输入处就已经很弱 ⇒ 加输出级损失救不回来，只能改数据")
    return 0


if __name__ == "__main__":
    sys.exit(main())
