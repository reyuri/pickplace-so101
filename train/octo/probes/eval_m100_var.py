# -*- coding: utf-8 -*-
"""单次采样 vs 多次平均：真机那 12.89 度的误差，是模型的锅还是「只抽一次」的锅？

部署一次只抽一个扩散样本(seed 42)就执行。如果单次误差 12.89 度、平均 9 次的误差
掉到 2 度以下，那模型本身没问题，是**采样噪声**在真机上乱动——改部署端取平均即可。
如果平均之后还是很差，那就是训练根本没学出来。

两个基线照旧：什么都不动 / 沿用上一帧。

用法
----
  $P eval_m100_var.py 8000        # 红框版
  $P eval_m100_var.py 8000 --plain
"""
from __future__ import annotations

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

TOYS = ("ball", "elephant", "tree")
VAL_LOCAL = (0, 9, 18, 27, 36)
FRAMES_PER_EP = 5
NSEED = 9
RED_TPL = "Pick up the %s in red box and place it into the box"
PLAIN_TPL = "Pick up the %s and place it into the box"


def to128(x):
    return np.asarray(Image.fromarray(np.array(x)).resize((128, 128)))


def main():
    step = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    plain = "--plain" in sys.argv
    root = "/root/autodl-tmp/octo_frames" if plain else "/root/autodl-tmp/octo_frames_redbox"
    ckpt = "/root/autodl-tmp/octo_lora_fix" if plain else "/root/autodl-tmp/octo_lora_redbox"
    tpl = PLAIN_TPL if plain else RED_TPL

    model = OctoModel.load_pretrained(ckpt, step=step)
    st = model.dataset_statistics["action"]
    amean = np.concatenate([np.asarray(st["mean"], np.float32), [0.0]])
    astd = np.concatenate([np.asarray(st["std"], np.float32), [1.0]])

    e1, em, ed, ep_ = [], [], [], []
    for toy in TOYS:
        for e in VAL_LOCAL:
            p = "%s/%s/ep_%d.npz" % (root, toy, e)
            if not os.path.exists(p):
                continue
            d = np.load(p)
            n = len(d["side"])
            task = {"language_instruction": model.text_processor.encode([tpl % toy])}
            for f in np.linspace(max(1, n // 5), n - 5, FRAMES_PER_EP).astype(int):
                obs = {"image_primary": jnp.asarray(d["side"][f - 1:f + 1][None], jnp.uint8),
                       "image_wrist": jnp.asarray(
                           np.stack([to128(x) for x in d["wrist"][f - 1:f + 1]])[None], jnp.uint8),
                       "timestep_pad_mask": jnp.ones((1, 2), dtype=bool)}
                draws = []
                for s in range(NSEED):
                    a = model.sample_actions(obs, task, argmax=False,
                                             rng=jax.random.PRNGKey(42 + s))
                    draws.append(np.asarray(a[0]))          # (4,7) normalized
                draws = np.stack(draws)                      # (K,4,7)
                gt = d["action"][f:f + 4, :6]
                k = len(gt)
                single = np.asarray(draws[0] * astd + amean)[:k, :6]
                mean = np.asarray(draws.mean(0) * astd + amean)[:k, :6]
                e1.extend(np.abs(single - gt))
                em.extend(np.abs(mean - gt))
                ed.extend(np.abs(d["state"][f][:6] - gt))
                ep_.extend(np.abs(d["action"][f - 1][:6] - gt))
            print("  跑完 %s/ep_%d" % (toy, e), flush=True)

    e1, em = np.asarray(e1), np.asarray(em)
    ed, ep_ = np.asarray(ed), np.asarray(ep_)
    NAMES = ["j0(肩旋)", "j1(肩)", "j2(肘)", "j3(腕俯)", "j4(腕旋)", "j5(夹爪)"]
    print()
    print("=" * 84)
    print("%s @ step=%d   val %d 帧  (toy,seed) 固定，只比「抽1次 vs 抽%d次取平均」" %
          ("原版(无框)" if plain else "红框版", step, len(e1) // 12, NSEED))
    print("-" * 84)
    print("  %-11s %11s %11s %13s %13s" % ("关节", "抽1次", "平均%d次" % NSEED,
                                           "「不动」", "「沿用上帧」"))
    for i, nm in enumerate(NAMES):
        print("  %-11s %11.2f %11.2f %13.2f %13.2f"
              % (nm, e1[:, i].mean(), em[:, i].mean(), ed[:, i].mean(), ep_[:, i].mean()))
    print("  %-11s %11.2f %11.2f %13.2f %13.2f"
          % ("全体", e1.mean(), em.mean(), ed.mean(), ep_.mean()))
    print("-" * 84)
    print("  单位 M100 度。平均后仍高于「不动」=> 模型本身没学出来，不是采样噪声的锅。")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    sys.exit(main())
