"""Task-sensitivity probe, parameterised so the baseline and the InfoNCE run can be
compared on byte-identical inputs.

  python probe_lang_cmp.py <ckpt_dir> <step> "<tag>"

Reports the first-step action under the three toy instructions.  The comparison is PAIRED
(same obs, same rng), so the shared diffusion noise cancels and only the instruction's
effect remains.  The unpaired control (same sentence, two different rngs) gives the scale
of what noise alone moves, as a reference for how big the sensitivity number is.
"""
import os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_CACHE", "/root/autodl-tmp/hf_hub")
# on-demand allocation: training is holding ~20 of the 24 GB on this card
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import glob, numpy as np, jax, jax.numpy as jnp
from octo.model.octo_model import OctoModel

TASKS = {
    "tree": "Pick up the tree and place it into the box",
    "ball": "Pick up the ball and place it into the box",
    "elephant": "Pick up the elephant and place it into the box",
}


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


d = np.load(sorted(glob.glob("/root/autodl-tmp/octo_frames/tree/ep_*.npz"))[0])
fidx = min(10, d["side"].shape[0] - 1)
i0 = max(0, fidx - 1)
obs = {
    "image_primary": jnp.asarray(np.stack([to_resize(d["side"][i0], (256, 256)),
                                           to_resize(d["side"][fidx], (256, 256))])[None], jnp.uint8),
    "image_wrist": jnp.asarray(np.stack([to_resize(d["wrist"][i0], (128, 128)),
                                         to_resize(d["wrist"][fidx], (128, 128))])[None], jnp.uint8),
    "timestep_pad_mask": jnp.ones((1, 2), dtype=bool),
}


def act(model, sentence, seed):
    t = {"language_instruction": model.text_processor.encode([sentence])}
    return np.asarray(model.sample_actions(obs, t, argmax=False,
                                           rng=jax.random.PRNGKey(seed))[0])


def main():
    ckpt, step, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    model = OctoModel.load_pretrained(ckpt, step=step)
    print("=" * 78, flush=True)
    print("%s   (%s / step %d)" % (tag, ckpt, step), flush=True)

    out = {k: act(model, s, 42) for k, s in TASKS.items()}
    for k in TASKS:
        print("   %-9s 首步 -> %s" % (k, np.round(out[k][0], 3)), flush=True)

    sens = max(np.abs(out[k] - out["tree"]).max() for k in ("ball", "elephant"))
    scale = max(np.abs(out[k]).max() for k in TASKS)
    # unpaired reference: identical sentence, two independent diffusion seeds
    noise = np.abs(act(model, TASKS["tree"], 1) - act(model, TASKS["tree"], 2)).max()

    print("   >> 任务敏感度 (归一化域) : %.4f" % sens, flush=True)
    print("   >> 动作幅度 scale        : %.4f" % scale, flush=True)
    print("   >> 噪声地板 (同句异rng)  : %.4f" % noise, flush=True)
    print("   >> 信噪比                : %.1fx" % (sens / max(noise, 1e-9)), flush=True)


if __name__ == "__main__":
    main()
