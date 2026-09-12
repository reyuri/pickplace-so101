# -*- coding: utf-8 -*-
"""VAL_loss 到底在量什么？把预测动作映射回 M100，看差多少度。

VAL_loss 的定义（octo/model/components/action_heads.py: DiffusionActionHead.loss）
------------------------------------------------------------------------------
  1. 从 [0, diffusion_steps) 随机抽一个噪声时刻 t
  2. 对**归一化后**的动作加噪:  x_t = sqrt(ᾱ_t)·a + sqrt(1-ᾱ_t)·ε
  3. 让网络预测 ε，loss = MSE(ε̂, ε)，再 **× action_dim(=7)**
  4. 在 eval 时对 30 个 batch 取平均

所以 VAL_loss 是「**噪声预测**的均方误差」，跑遍所有噪声档位后取平均，
单位是归一化空间，**不是角度**，也不能直接读成「偏了几度」。它主要被高噪声档位主导。

要回答「M100 差多少」，只能自己采样再反归一化来量——就是本脚本做的事。

口径与部署一致：2 帧窗口、纯视觉(无 proprio)、argmax=False、seed 42（同 serve）。

用法
----
  P=/root/autodl-tmp/conda_envs/octo/bin/python
  $P eval_m100.py 8000            # 红框版 val 集
  $P eval_m100.py 8000 --plain    # 同 step 的原版(无框)做参照
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


def to128(x):
    return np.asarray(Image.fromarray(np.array(x)).resize((128, 128)))

TOYS = ("ball", "elephant", "tree")          # 与 builder 的 sorted() 一致
VAL_LOCAL = (0, 9, 18, 27, 36)               # ep % 9 == 0，每 toy 5 条
FRAMES_PER_EP = 8
RED_TPL = "Pick up the %s in red box and place it into the box"
PLAIN_TPL = "Pick up the %s and place it into the box"


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

    errs, errs_do, errs_prev, chunk_errs = [], [], [], []
    for toy in TOYS:
        for ep in VAL_LOCAL:
            p = "%s/%s/ep_%d.npz" % (root, toy, ep)
            if not os.path.exists(p):
                print("[skip] %s" % p, flush=True)
                continue
            d = np.load(p)
            n = len(d["side"])
            # f-1 要存在(窗口是 [f-1,f])，f+4 也要存在(chunk)
            picks = np.linspace(1, n - 5, FRAMES_PER_EP).astype(int)
            task = {"language_instruction": model.text_processor.encode([tpl % toy])}
            for f in picks:
                obs = {
                    "image_primary": jnp.asarray(d["side"][f - 1:f + 1][None], jnp.uint8),
                    "image_wrist": jnp.asarray(
                        np.stack([to128(x) for x in d["wrist"][f - 1:f + 1]])[None], jnp.uint8),
                    "timestep_pad_mask": jnp.ones((1, 2), dtype=bool),
                }
                a = model.sample_actions(obs, task, argmax=False,
                                         rng=jax.random.PRNGKey(42))
                pred = np.asarray(a[0] * astd + amean)[:, :6]        # (4,6) M100
                gt = d["action"][f:f + 4, :6]                        # (4,6) M100
                k = len(gt)
                errs.extend(np.abs(pred[:k] - gt))
                errs_do.extend(np.abs(d["state"][f][:6] - gt))        # 什么都不动
                errs_prev.extend(np.abs(d["action"][f - 1][:6] - gt))  # 沿用上一帧动作
                chunk_errs.append(float(np.abs(pred[:k] - gt).mean()))
            print("  跑完 %s/ep_%d  (%d 帧)" % (toy, ep, len(picks)), flush=True)

    errs = np.asarray(errs); errs_do = np.asarray(errs_do); errs_prev = np.asarray(errs_prev)
    NAMES = ["j0(肩旋)", "j1(肩)", "j2(肘)", "j3(腕俯)", "j4(腕旋)", "j5(夹爪)"]
    print()
    print("=" * 78)
    print("%s @ step=%d    val 集 %d 帧 x 4 步 chunk" %
          ("原版(无框)" if plain else "红框版", step, len(chunk_errs)))
    print("-" * 78)
    print("  %-12s %10s %12s %12s" % ("关节", "模型 MAE", "「不动」MAE", "「沿用上一帧」"))
    for i, nm in enumerate(NAMES):
        print("  %-12s %10.2f %12.2f %12.2f" % (nm, errs[:, i].mean(),
                                                errs_do[:, i].mean(), errs_prev[:, i].mean()))
    print("  %-12s %10.2f %12.2f %12.2f" % ("全体", errs.mean(), errs_do.mean(), errs_prev.mean()))
    print("-" * 78)
    print("  单位 = M100 度。模型必须**低于**两个惰性基线才算真的学到了动作；")
    print("  低于多少 = 训练到底做没做出来。")
    print("  chunk 内平均 |误差| = %.2f 度" % float(np.mean(chunk_errs)))
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
