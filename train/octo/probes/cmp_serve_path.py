"""Same frame, same denorm+clip as serve_octo_native, for any checkpoint."""
import sys, glob
import numpy as np, jax, jax.numpy as jnp
from PIL import Image
from octo.model.octo_model import OctoModel

TASKS = {"tree": "Pick up the tree and place it into the box",
         "ball": "Pick up the ball and place it into the box",
         "elephant": "Pick up the elephant and place it into the box"}
ckpt, step, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
m = OctoModel.load_pretrained(ckpt, step=step)
st = m.dataset_statistics["action"]
amean, astd = np.asarray(st["mean"])[:6], np.asarray(st["std"])[:6]

d = np.load(sorted(glob.glob("/root/autodl-tmp/octo_frames/tree/ep_*.npz"))[0])
f = min(10, d["side"].shape[0] - 1)
def prep(img, t):
    a = np.ascontiguousarray(img[:, :, :3].astype(np.uint8))
    return np.ascontiguousarray(np.asarray(Image.fromarray(a).resize((t, t), Image.LANCZOS)))
side, wrist = prep(d["side"][f], 256), prep(d["wrist"][f], 128)
obs = {"image_primary": jnp.asarray(np.stack([side, side])[None], jnp.uint8),
       "image_wrist":   jnp.asarray(np.stack([wrist, wrist])[None], jnp.uint8),
       "timestep_pad_mask": jnp.ones((1, 2), dtype=bool)}
print("=" * 74); print("%s  (%s / step %d)" % (tag, ckpt, step))
out = {}
for k, t in TASKS.items():
    raw = np.asarray(m.sample_actions(obs, {"language_instruction": m.text_processor.encode([t])},
                                      argmax=False, rng=jax.random.PRNGKey(42))[0])
    a = np.clip(raw[:, :6] * astd + amean, -100.0, 100.0)
    out[k] = a
    print("  %-9s norm首步=%s" % (k, np.round(raw[0, :6], 2)))
    print("            M100首步=%s" % np.round(a[0], 2))
print("  两两首步最大差: t-b=%.2f  t-e=%.2f  b-e=%.2f"
      % (np.abs(out["tree"][0]-out["ball"][0]).max(), np.abs(out["tree"][0]-out["elephant"][0]).max(),
         np.abs(out["ball"][0]-out["elephant"][0]).max()))
print("  整块最大差=%.2f   max|a|=%.2f" % (max(np.abs(out[a]-out[b]).max() for a in out for b in out),
                                            max(np.abs(v).max() for v in out.values())))
