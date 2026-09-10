"""Fixed-dimension, multi-seed language-sensitivity probe.

probe_lang_clean.py excluded "saturated" dims, but the exclusion mask was recomputed PER
CHECKPOINT -- step 2000 compared dims [2,6], step 6000 compared [1,2,5,6], steps 4000/8000
compared all seven.  Different denominators are not comparable, so that aggregate meant
nothing.  This version pins the comparison to ALL SEVEN dims always, and reports the
saturation mask only as a side note.

It also fixes the other weakness: each estimate came from a single diffusion seed, and the
noise floor IS diffusion sampling variance, so it swung 1.0 -> 5.4 between checkpoints.
Here every frame is sampled with several seeds; sensitivity is averaged over seed-paired
instruction swaps (shared noise cancels) and the noise floor over several independent-seed
pairs of the SAME instruction.

Primary readout, comparable across every checkpoint and the baseline:

    RATIO = mean_over_7_dims(sens) / mean_over_7_dims(noise)

Usage: python probe_lang_fixeddim.py <ckpt_dir> <step> "<tag>"
"""
import os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_CACHE", "/root/autodl-tmp/hf_hub")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import glob, itertools, numpy as np, jax, jax.numpy as jnp
from octo.model.octo_model import OctoModel

TASKS = {
    "tree": "Pick up the tree and place it into the box",
    "ball": "Pick up the ball and place it into the box",
    "elephant": "Pick up the elephant and place it into the box",
}
SEEDS = [42, 43, 44, 45]
FRAMES = [10, 40, 70]
RAIL = 4.9
D = 7


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


def main():
    ckpt, step, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    d = np.load(sorted(glob.glob("/root/autodl-tmp/octo_frames/tree/ep_*.npz"))[0])
    n = d["side"].shape[0]
    model = OctoModel.load_pretrained(ckpt, step=step)
    print("=" * 78, flush=True)
    print("%s   (%s / step %d)" % (tag, ckpt, step), flush=True)

    def act(obs, sentence, seed):
        t = {"language_instruction": model.text_processor.encode([sentence])}
        return np.asarray(model.sample_actions(obs, t, argmax=False,
                                               rng=jax.random.PRNGKey(seed))[0])  # (H, D)

    sens_f, noise_f, peaks = [], [], []
    for f in FRAMES:
        f = min(f, n - 1)
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
        o = {(s, k): act(obs, sen, s) for s in SEEDS for k, sen in TASKS.items()}
        # paired: same seed, different instruction -> shared diffusion noise cancels
        sens_f.append(np.mean([np.abs(o[(s, k)] - o[(s, "tree")])
                               for s in SEEDS for k in ("ball", "elephant")], axis=(0, 1)))
        # unpaired: same instruction, independent seeds -> pure sampling variance
        noise_f.append(np.mean([np.abs(o[(a, "tree")] - o[(b, "tree")])
                                for a, b in itertools.combinations(SEEDS, 2)], axis=(0, 1)))
        peaks.append(np.max(np.abs(np.stack([o[v] for v in o])), axis=(0, 1)))

    S, N, P = np.array(sens_f).mean(0), np.array(noise_f).mean(0), np.array(peaks).max(0)
    Se = np.array(sens_f).std(0) / np.sqrt(len(FRAMES))
    Ne = np.array(noise_f).std(0) / np.sqrt(len(FRAMES))
    print("     dim   sens(+-se)     noise(+-se)    ratio    peak", flush=True)
    for i in range(D):
        print("      %d   %6.3f+-%.3f   %6.3f+-%.3f   %5.2fx  %6.3f%s"
              % (i, S[i], Se[i], N[i], Ne[i], S[i] / max(N[i], 1e-9), P[i],
                 "  <-ON RAIL" if P[i] >= RAIL else ""), flush=True)
    print("   >>> FIXED 7-DIM RATIO = %.3fx   (sens=%.4f  noise=%.4f)"
          % (S.mean() / max(N.mean(), 1e-9), S.mean(), N.mean()), flush=True)
    print("   (all-frame max |a| = %.3f; >= %.1f means that dim sits on the clip rail)"
          % (P.max(), RAIL), flush=True)


if __name__ == "__main__":
    main()
