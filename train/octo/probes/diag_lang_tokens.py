# -*- coding: utf-8 -*-
"""第二步诊断：语言通路是"死了"，还是"只认句子、不认名词"？

上一步（diag_lang_layers.py）的结论：
  * 动作头不是元凶 —— readout 与 action 的 S 同量级（0.13 vs 0.14）
  * 语言在 readout 处就已经很弱：换指令 Δ=1.58%，换空串 Δ=1.00%
  * 即"真实指令"和"没有指令"对 readout 的扰动差不多大

但 1.58% 这个数还分不出两种完全不同的病因，这个探针就是来分的：

  (甲) 语言通路根本没接上 —— cross-attention 没学到去读语言 token。
       那么"换名词"和"换整句"应该**一样弱**，模型对文本内容整体无感。

  (乙) 通路是通的，只是分辨不出**那一个名词**。
       三句指令只差 tree/ball/elephant 一个词，其余 8 个词完全相同；
       frozen T5 + 只训 Q/V LoRA，很可能没学会把注意力压到那一个 token 上。
       那么"换整句"应该**明显强于**"换名词"。

这两者的修法完全不同：
  甲 ⇒ 得改语言接口本身（解冻更多层 / 换 pooling / 重建 task 串）
  乙 ⇒ 得让名词 token 被看见（提示词重写、token 级加权、或名词对比损失）

对动作层同样测量，但动作层采样噪声大，只作旁证，结论以 readout 为准。
动作层一律用「同帧同 seed 配对」，让扩散噪声共模抵消。

用法：
  P=/root/autodl-tmp/conda_envs/octo/bin/python
  $P diag_lang_tokens.py /root/autodl-tmp/octo_lora_fix 12000
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

# 每一类都对着 tree 比。括号里是"和 tree 差在哪"。
TASKS = {
    # ---- 只差一个名词，其余 8 个词完全相同 ----
    "ball":       ("Pick up the ball and place it into the box",       "只换名词"),
    "elephant":   ("Pick up the elephant and place it into the box",   "只换名词"),
    # ---- 整句都不同 ----
    "other1":     ("Throw the ball away immediately",                  "换整句"),
    "other2":     ("Move the arm slowly to the left",                  "换整句"),
    # ---- 边界情况 ----
    "empty":      ("",                                                 "无语言"),
    "gibberish":  ("Banana helicopter purple",                         "胡言乱语"),
}
REF = "tree"
REF_SENT = "Pick up the tree and place it into the box"


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
    frames = []
    for toy in ("tree", "ball", "elephant"):
        for p in sorted(glob.glob(os.path.join(FRAMES_ROOT, toy, "ep_*.npz")))[:FRAMES_PER_TOY]:
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
            frames.append(obs)
    return frames


def rel(u, v):
    return float(np.linalg.norm(u - v) / (np.linalg.norm(v) + 1e-12))


def main() -> int:
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/octo_lora_fix"
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 12000
    model = OctoModel.load_pretrained(ckpt, step=step)
    bm = model.module.bind({"params": model.params},
                           rngs={"dropout": jax.random.PRNGKey(0)})
    frames = load_frames()
    nf = len(frames)

    # --- 先看一眼 T5 侧：名词 token 的嵌入到底有多不一样（纯净的语言输入） ---
    print("=" * 88)
    print("T5 输入的 token 级差异（对照：如果这里都很小，那问题不在 transformer）")
    print("-" * 88)
    enc_ref = model.text_processor.encode([REF_SENT])
    ids_ref = np.asarray(enc_ref["input_ids"][0])
    mask_ref = np.asarray(enc_ref["attention_mask"][0])
    print("  REF ids = %s" % ids_ref.tolist())
    for k, (s, kind) in TASKS.items():
        e = model.text_processor.encode([s])
        ids = np.asarray(e["input_ids"][0])
        mask = np.asarray(e["attention_mask"][0])
        n = min(len(ids), len(ids_ref))
        same = int((ids[:n] == ids_ref[:n]).sum())
        print("  %-10s len=%2d  mask_sum=%2d  与 REF 同位相同的 token 数 = %d/%d"
              % (k, len(ids), int(mask.sum()), same, n))
    print("  （3 句只换名词的指令，除名词外应几乎全同；这就是「语言冗余」的物质基础）")

    # --- 逐帧 × 逐指令 ---
    Z, A = {}, {}
    for fi in range(nf):
        for k, (s, _) in TASKS.items():
            cur = s
            tsk = {"language_instruction": model.text_processor.encode([cur])}
            emb = bm.octo_transformer(frames[fi], tsk, frames[fi]["timestep_pad_mask"], train=False)
            Z[(fi, k)] = np.asarray(emb["readout_action"].tokens).reshape(-1).astype(np.float64)
            for sd in SEEDS:
                a = model.sample_actions(frames[fi], tsk, argmax=False,
                                         rng=jax.random.PRNGKey(1000 + sd))
                A[(fi, k, sd)] = np.asarray(a[0]).astype(np.float64)
        print("  [帧 %d/%d] 跑完" % (fi + 1, nf), flush=True)

    # REF 也要算（放在最后，避免和上面的循环纠缠）
    for fi in range(nf):
        tsk = {"language_instruction": model.text_processor.encode([REF_SENT])}
        emb = bm.octo_transformer(frames[fi], tsk, frames[fi]["timestep_pad_mask"], train=False)
        Z[(fi, REF)] = np.asarray(emb["readout_action"].tokens).reshape(-1).astype(np.float64)
        for sd in SEEDS:
            a = model.sample_actions(frames[fi], tsk, argmax=False,
                                     rng=jax.random.PRNGKey(1000 + sd))
            A[(fi, REF, sd)] = np.asarray(a[0]).astype(np.float64)

    print()
    print("=" * 88)
    print("相对变化量（对 REF「Pick up the tree...」）—— 结论看 readout 列")
    print("-" * 88)
    print("  %-11s %-8s %16s %16s" % ("指令", "类型", "readout ΔL", "动作 ΔL(配对)"))
    rows = {}
    for k, (s, kind) in TASKS.items():
        dz = float(np.mean([rel(Z[(fi, k)], Z[(fi, REF)]) for fi in range(nf)]))
        da = float(np.mean([rel(A[(fi, k, sd)], A[(fi, REF, sd)])
                            for fi in range(nf) for sd in SEEDS]))
        rows[k] = (kind, dz, da)
        print("  %-11s %-8s %16.4f %16.4f" % (k, kind, dz, da))

    noun = [rows["ball"][1], rows["elephant"][1]]
    sent = [rows["other1"][1], rows["other2"][1]]
    mn, ms = float(np.mean(noun)), float(np.mean(sent))
    print("-" * 88)
    print("  换名词  平均 readout ΔL = %.4f" % mn)
    print("  换整句  平均 readout ΔL = %.4f" % ms)
    print()
    if ms < mn * 1.5:
        print("  判定 (甲)：换整句 ≈ 换名词（比值 %.2f）⇒ **语言通路基本没接通**，"
              % (ms / max(mn, 1e-12)))
        print("            模型对文本内容整体不敏感，不是「认不出那个名词」的问题。")
        print("            ⇒ readout/输出级的对比损失都是无本之木：通路本身没建立，")
        print("              损失只能逼出一个模型从未学会区分的差异。")
        print("              要修就得动语言接口（解冻层数 / pooling / task 串构造）。")
    elif ms > mn * 3:
        print("  判定 (乙)：换整句 >> 换名词（比值 %.2f）⇒ **通路是通的，但只认句子不认名词**。"
              % (ms / max(mn, 1e-12)))
        print("            ⇒ 病灶是三句指令只差一个 token，注意力没压到那个 token 上。")
        print("              这有得救：提示词重写 / 名词 token 加权 / 名词级对比损失都可试。")
    else:
        print("  判定：换整句略强于换名词（比值 %.2f），倾向 (乙) 但不显著。" % (ms / max(mn, 1e-12)))
        print("        ⇒ 通路部分可用，名词分辨是主要瓶颈。")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    sys.exit(main())
