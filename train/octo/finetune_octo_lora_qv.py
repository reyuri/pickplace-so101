"""Octo finetune — Q/V LoRA + full-MLP + full-diffusion-head (head_mlp_only style).

Same data/shape as the native pivot (vision-only, window2, primary256/wrist128,
horizon4, action 6->7 pad) but, per the locked config:
  - transformer_kwargs.lora_rank=8  -> LoRA adapters ONLY on attention query/value
    (fork's transformer.py builds ``*_lora_down/up`` on query & value only; key/out
    and MLP have NO LoRA, so base kernels stay).
  - TRANSFORMER MLP (MlpBlock Dense_0/Dense_1) unfrozen -> trained fully.
  - DiffusionActionHead (pretrained dim7/horizon4, merged in) unfrozen -> trained fully.
  - ViT (primary/wrist) + T5 (hf_model) + attention base kernels + LayerNorm frozen.

`--smoke` builds the model, prints the trainable/frozen partition (with representative
leaf paths), then runs ~2 steps asserting (a) some params actually change and (b) the
FROZEN subset is byte-identical across steps — i.e. this is NOT a no-op.
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_CACHE", "/root/autodl-tmp/hf_hub")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from absl import app, flags
import numpy as np
import jax, jax.numpy as jnp, optax, tensorflow as tf, tqdm
tensorflow_datasets = tfds = __import__("tensorflow_datasets")
from lerobot_toy_builder.lerobot_toy_dataset import LerobotToy
from octo.data.dataset import make_single_dataset
import dlimp as _dlimp
from tensorflow_datasets.core.utils import gcs_utils
gcs_utils._is_gcs_disabled = True
from octo.model.octo_model import OctoModel
from octo.utils.train_utils import merge_params, process_text, TrainState
from octo.utils.jax_utils import initialize_compilation_cache

FLAGS = flags.FLAGS
flags.DEFINE_string("pretrained_path", "/root/autodl-tmp/octo_weights", "Pretrained Octo (base).")
flags.DEFINE_string("data_dir", "/root/autodl-tmp/octo_tfds", "RLDS data dir.")
flags.DEFINE_string("save_dir", "/root/autodl-tmp/octo_lora_qv_out", "Checkpoint save dir.")
flags.DEFINE_integer("batch_size", 8, "Batch size.")
flags.DEFINE_integer("max_steps", 30000, "Number of finetuning steps.")
flags.DEFINE_integer("eval_freq", 1000, "Frequency to compute val loss + save.")
flags.DEFINE_integer("val_batches", 30, "Number of val batches to average.")
flags.DEFINE_integer("lora_rank", 8, "LoRA rank (input to transformer_kwargs).")
flags.DEFINE_float("lora_scale", 1.0, "LoRA scale.")
flags.DEFINE_float("lr", 1e-4, "AdamW LR (trainable partition).")
flags.DEFINE_boolean("smoke", False, "Build + print partition + run 2 steps, no long train.")
flags.DEFINE_integer("step", None, "Specific checkpoint step to load (None=auto-latest).")
flags.DEFINE_float("infonce_w", 0.0,
                   "Weight of the readout-embedding InfoNCE term. 0 = off (= original).")
flags.DEFINE_float("infonce_tau", 0.5, "Temperature for the InfoNCE logsumexp.")
flags.DEFINE_float("infonce_margin", 0.3,
                   "TARGET cosine between the K readout variants for different instructions. "
                   "The loss is a squared well around it: above the target the variants are "
                   "pushed apart, below it they are pulled back, and cos -> -1 is penalised "
                   "too. Unlike a hinge this pressure never switches off.")
flags.DEFINE_integer("infonce_neg", 1,
                     "Counterfactual instructions per step (K = 1 + this). 2 needs a smaller "
                     "--batch_size: the K-tiled forward scales backward memory by K.")

# Transformer (octo_transformer) MLP block — scope so ViT/T5 blocks are excluded.
_TRANSFORMER_MARKERS = ("octo_transformer", "Transformer_", "encoderblock_", "encoder_norm")
_FROZEN_REGIONS = ("observation_tokenizers", "hf_model", "hl_model", "vf_model", "InputLayer")


def _is_trainable(path):
    s = ".".join(str(p.key) for p in path)
    if "_lora" in s:
        return True                                   # attention query/value LoRA adapters
    if s.startswith("heads"):
        return True                                   # DiffusionActionHead (score-net/readout)
    if "MlpBlock" in s and any(m in s for m in _TRANSFORMER_MARKERS):
        if not any(f in s for f in _FROZEN_REGIONS):
            return True                               # transformer MLP (Dense_0/Dense_1)
    if ("task_language_projection" in s or "task_language_pos_embedding" in s) and \
            not any(f in s for f in _FROZEN_REGIONS):
        # The frozen interface between T5 and the transformer: unfreezing it lets the model
        # re-ENCODE the instruction instead of only re-routing a fixed encoding through the
        # Q/V LoRA.  ~603k params.  T5 itself (task_tokenizers_language.hf_model) matches
        # _FROZEN_REGIONS and stays frozen -- no catastrophic forgetting of the encoder.
        return True
    return False


def _patch_tfds_builder(data_dir):
    _orig = tfds.builder
    def patched(name, data_dir_=None, **kw):
        if str(name).split(":")[0] == "lerobot_toy":
            return LerobotToy(data_dir=data_dir_ or data_dir)
        return _orig(name, data_dir=data_dir_, **kw)
    tfds.builder = patched

def _patch_dlimp_from_rlds():
    _orig = _dlimp.DLataset.from_rlds
    def patched(builder, split="train", shuffle=True, num_parallel_reads=tf.data.AUTOTUNE):
        oa = builder.as_dataset
        def no_skip(*a, **kw):
            kw.pop("decoders", None); return oa(*a, **kw)
        builder.as_dataset = no_skip
        return _orig(builder, split=split, shuffle=shuffle, num_parallel_reads=num_parallel_reads)
    _dlimp.DLataset.from_rlds = patched

def _make_dataset(train, statistics=None):
    return make_single_dataset(
        dataset_kwargs=dict(name="lerobot_toy", data_dir=FLAGS.data_dir,
            image_obs_keys={"primary": "side", "wrist": "eye_in_hand"},
            language_key="language_instruction", dataset_statistics=statistics),
        traj_transform_kwargs=dict(window_size=2, action_horizon=4),
        frame_transform_kwargs=dict(resize_size={"primary": (256, 256), "wrist": (128, 128)}),
        train=train)

def pad_action7(x):
    if x.shape[-1] != 6:
        return x
    return jnp.concatenate([x, jnp.zeros_like(x[..., :1])], axis=-1)

TOYS = ("elephant", "tree", "ball")

def wrong_instruction(s, offset):
    """Swap the toy named in the instruction for a different one (template agnostic)."""
    hits = [t for t in TOYS if t in s]
    if len(hits) != 1:
        raise ValueError(
            "cannot derive a counterfactual instruction from %r: expected exactly one of "
            "%s to appear." % (s, TOYS))
    t = hits[0]
    return s.replace(t, TOYS[(TOYS.index(t) + offset) % 3])

def main(_):
    _patch_tfds_builder(FLAGS.data_dir)
    _patch_dlimp_from_rlds()
    initialize_compilation_cache()
    pretrained_model = OctoModel.load_pretrained(FLAGS.pretrained_path, step=FLAGS.step)

    train_dataset = _make_dataset(train=True)
    train_stats = train_dataset.dataset_statistics
    val_dataset = _make_dataset(train=False, statistics=train_stats)

    text_processor = pretrained_model.text_processor
    def process_batch(batch):
        # capture the raw sentences BEFORE process_text overwrites them with token ids
        raw = [s.decode("utf-8") for s in batch["task"]["language_instruction"]]
        batch = process_text(batch, text_processor)
        del batch["dataset_name"]
        if "action" in batch:
            batch["action"] = pad_action7(batch["action"])
        if "action_pad_mask" in batch:
            batch["action_pad_mask"] = pad_action7(batch["action_pad_mask"])
        if FLAGS.infonce_w > 0:
            # the other two toys, i.e. the counterfactual language variants of this frame.
            # Mirror batch["task"] exactly ({language_instruction, pad_mask_dict}) and only
            # swap the encoded instruction, so the pytree structure matches under tree_map.
            for off in (1, 2):
                w = dict(batch["task"])
                w["language_instruction"] = text_processor.encode(
                    [wrong_instruction(s, off) for s in raw])
                batch["task_wrong%d" % off] = w
        return batch
    train_iter = map(process_batch, (train_dataset.repeat().unbatch().shuffle(10000).batch(FLAGS.batch_size).iterator()))
    val_iter = map(process_batch, (val_dataset.repeat().unbatch().batch(FLAGS.batch_size).iterator()))

    example_batch = next(train_iter)
    cfg = pretrained_model.config
    cfg["model"]["transformer_kwargs"] = dict(cfg["model"].get("transformer_kwargs", {}))
    cfg["model"]["transformer_kwargs"]["lora_rank"] = FLAGS.lora_rank
    cfg["model"]["transformer_kwargs"]["lora_scale"] = FLAGS.lora_scale
    model = OctoModel.from_config(cfg, example_batch, text_processor,
                                  verbose=True, dataset_statistics=train_stats)
    merged = merge_params(model.params, pretrained_model.params)
    model = model.replace(params=merged); del pretrained_model

    # --- partition: trainable vs frozen ---
    def lora_mask(params):
        return jax.tree_util.tree_map_with_path(
            lambda path, x: "train" if _is_trainable(path) else "frozen", params)
    mask = lora_mask(model.params)

    # categorize + count (+ representative paths) for --smoke
    counts = {"train": 0, "frozen": 0}
    reps = {"train": [], "frozen": []}
    for path, m in jax.tree_util.tree_flatten_with_path(mask)[0]:
        counts[m] += 1
        s = ".".join(str(p.key) for p in path)
        if m == "train":
            if len(reps["train"]) < 25:
                reps["train"].append(s)
        elif len(reps["frozen"]) < 25:
            reps["frozen"].append(s)
    print(f"[partition] train leaves={counts['train']}  frozen leaves={counts['frozen']}", flush=True)
    print("[train reps]", flush=True)
    for s in reps["train"]:
        print("   ", s, flush=True)
    print("[frozen reps]", flush=True)
    for s in reps["frozen"]:
        print("   ", s, flush=True)

    # diagnostic: what's inside train / any leak into frozen regions
    from collections import Counter
    acats = Counter()
    leak = 0
    for path, m in jax.tree_util.tree_flatten_with_path(mask)[0]:
        s = ".".join(str(p.key) for p in path)
        if m == "train":
            if "_lora" in s:
                acats["lora"] += 1
            elif s.startswith("heads"):
                acats["head"] += 1
            elif "MlpBlock" in s:
                acats["mlp"] += 1
            else:
                acats["train_other"] += 1
            if any(f in s for f in _FROZEN_REGIONS):
                leak += 1
    print(f"[diag] train breakdown={dict(acats)}  train_leaves_in_frozen_region={leak}", flush=True)

    # --- optimizer: trainable -> adamw(lr), frozen -> no update ---
    lr_sched = optax.join_schedules(
        [optax.linear_schedule(0, FLAGS.lr, 500), optax.linear_schedule(FLAGS.lr, 0, FLAGS.max_steps)], [500])
    tx = optax.multi_transform({"train": optax.adamw(lr_sched), "frozen": optax.set_to_zero()}, mask)
    train_state = TrainState.create(rng=jax.random.PRNGKey(2026), model=model, tx=tx)

    def loss_fn(params, batch, task_wrongs, rng, train=True):
        bm = model.module.bind({"params": params}, rngs={"dropout": rng})
        obs = batch["observation"]
        tpm = obs["timestep_pad_mask"]
        emb = bm.octo_transformer(obs, batch["task"], tpm, train=train)
        loss, met = bm.heads["action"].loss(emb, batch["action"], tpm,
                                            batch["action_pad_mask"], train=train)
        if FLAGS.infonce_w <= 0 or task_wrongs is None:
            return loss, met

        # --- InfoNCE over the K language variants of the SAME frame ------------------
        # K = 1 correct + (K-1) counterfactuals.  All K share one image, so this term can
        # only be reduced by making the readout embedding depend on the instruction --
        # exactly the supervision the plain diffusion MSE lacks (vision alone already
        # drives its loss down, so the language path gets no gradient).
        #
        # Note on the positive term: with one shared image, cos(e_k, e_k) == 1 for every
        # variant, so InfoNCE's positive term is a CONSTANT and drops out.  What remains
        # is the logsumexp over the counterfactual variants.  Written as the usual
        # -log softmax it would saturate to ~0 as soon as the off-diagonal cosines fell
        # below ~0.5 and stop producing gradient; this form keeps descending instead.
        K = 1 + len(task_wrongs)
        B = tpm.shape[0]
        tile = lambda x, k: jnp.tile(x, (k,) + (1,) * (x.ndim - 1))
        # order is [variant0 (B), variant1 (B), ...] so a row's variant is index // B
        obs_k = jax.tree_util.tree_map(lambda x: tile(x, K), obs)
        task_k = jax.tree_util.tree_map(
            lambda *xs: jnp.concatenate(list(xs), axis=0), batch["task"], *task_wrongs)
        # Memory: the K-tiled forward multiplies the backward activation memory by K
        # (f32[batch*K,12,690,690] attention buffers per block -- 17 GiB temp at K=3, OOM on
        # this 24GB card).  jax.checkpoint cannot wrap a Flax module call (JaxTransformError),
        # so instead keep K small: --infonce_neg 1 contrasts against ONE counterfactual,
        # alternating between the other two toys across steps.
        emb_k = bm.octo_transformer(obs_k, task_k, tile(tpm, K), train=train)

        # ---- per-TOKEN contrast (see patch_fix_einsum.py for the two reasons) -------------
        # tokens is (K*B, W, T, D) with row i = variant (i // B), batch (i % B).
        z = emb_k["readout_action"].tokens                        # (K*B, W, T, D)
        W, T = z.shape[1], z.shape[2]
        z = z.reshape(K, B, W * T, -1).transpose(1, 2, 0, 3)      # (B, W*T, K, D)
        z = z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-6)
        # z is (B, W*T, K, D); the FEATURE axis (f) is shared and summed.  Writing a
        # subscript that gives the two operands different feature letters would repeat
        # the original bug -- it would sum them independently and take an outer product.
        sim = jnp.einsum("bmkf,bmlf->bmkl", z, z)                 # (B, W*T, K, K)

        off = ~jnp.eye(K, dtype=bool)
        n_off = B * W * T * K * (K - 1)
        off_sim = jnp.where(off[None, None], sim, 0.0)
        # WELL, not relu and not logsumexp -- both previous objectives failed:
        #   * logsumexp (K=2 => sim/tau) is unbounded below: it drives cos to -1, ANTI-parallel
        #     readouts, and wrecked the action loss at --infonce_w 20.
        #   * relu(cos - margin) fixed the runaway but RELEASES PERMANENTLY once satisfied, so
        #     the action loss -- which has no incentive to use language on this redundant data
        #     -- pulled the readout straight back.  Measured: SNR 1.7x at step 1000, 0.9x at
        #     2000, i.e. back under the noise floor.
        # The squared well has neither failure mode and the pressure never switches off.
        err = jnp.where(off[None, None], off_sim - FLAGS.infonce_margin, 0.0)
        infonce = jnp.sum(err ** 2) / n_off
        # mean pairwise cosine between the K variants, computed PER TOKEN: 1.0 == language has
        # no effect on the readout at all.  This is the number to watch during training.
        # NOT comparable to the 0.284 -> 0.00005 printed by the first run: that came from the
        # broken subscript and measured something else entirely.
        offdiag_cos = jnp.sum(off_sim) / n_off

        met = dict(met)
        met["action_loss"] = loss
        met["infonce"] = infonce
        met["offdiag_cos"] = offdiag_cos
        met["loss"] = loss + FLAGS.infonce_w * infonce
        return loss + FLAGS.infonce_w * infonce, met

    @jax.jit
    def train_step(state, batch, task_wrongs):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.model.params, batch, task_wrongs, dropout_rng, train=True)
        return state.apply_gradients(grads=grads, rng=rng), info

    @jax.jit
    def eval_step(state, batch, task_wrongs):
        _, info = loss_fn(state.model.params, batch, task_wrongs, state.rng, train=False)
        return info

    # --- smoke: verify trainable moves / frozen doesn't ---
    gather = jax.jit(lambda p: jax.tree_util.tree_leaves(p))

    def check_partition_moved(s0, s1):
        moved = 0
        kept = 0
        for p0, p1, m in zip(gather(s0), gather(s1), mask_flattened):
            same = bool(jnp.array_equal(p0, p1))
            if m == "train" and not same:
                moved += 1
            elif m == "frozen" and same:
                kept += 1
        return moved, kept

    if FLAGS.smoke:
        mask_flattened = jax.tree_util.tree_leaves(mask)
        p0 = gather(train_state.model.params)
        b = next(train_iter)
        if FLAGS.infonce_w > 0:
            # baseline language sensitivity of the STARTING model, before any step
            _, m0 = loss_fn(train_state.model.params, b, (b["task_wrong1"],),
                            train_state.rng, train=False)
            m0 = jax.device_get(m0)
            print("[infonce] start offdiag_cos=%.6f (1.0 == language has no effect)  "
                  "infonce=%.6f  action_loss=%.6f" %
                  (float(m0["offdiag_cos"]), float(m0["infonce"]), float(m0["action_loss"])),
                  flush=True)
        tw = (b["task_wrong1"],) if FLAGS.infonce_w > 0 else None
        train_state, info = train_step(train_state, b, tw)
        b2 = next(train_iter)
        tw2 = (b2["task_wrong2"],) if FLAGS.infonce_w > 0 else None
        train_state, i2 = train_step(train_state, b2, tw2)
        info = jax.device_get(info); i2 = jax.device_get(i2)
        p1 = gather(train_state.model.params)
        moved, kept = check_partition_moved(p0, p1)
        tot = len(mask_flattened)
        print(f"[smoke] loss@1={float(info['loss']):.4f}  loss@2={float(i2['loss']):.4f}", flush=True)
        print(f"[smoke] train leaves that MOVED: {moved} / {counts['train']}", flush=True)
        print(f"[smoke] frozen leaves still byte-identical: {kept} / {counts['frozen']}", flush=True)
        return

    os.makedirs(FLAGS.save_dir, exist_ok=True)
    for i in tqdm.tqdm(range(FLAGS.max_steps), total=FLAGS.max_steps, dynamic_ncols=True):
        batch = next(train_iter)
        if FLAGS.infonce_w > 0:
            w = [batch.pop("task_wrong1"), batch.pop("task_wrong2")]
            # alternate which counterfactual is contrasted against, so both ordered pairs
            # (A->B and A->C) still get covered across steps
            task_wrongs = tuple(w) if FLAGS.infonce_neg >= 2 else (w[np.random.randint(0, 2)],)
        else:
            task_wrongs = None
        train_state, info = train_step(train_state, batch, task_wrongs)
        step = i + 1
        if step % 100 == 0:
            info = jax.device_get(info)
            if FLAGS.infonce_w > 0:
                print(f"[step {step}] loss={float(info['loss']):.4f} "
                      f"action={float(info['action_loss']):.4f} "
                      f"infonce={float(info['infonce']):.4f} "
                      f"offdiag_cos={float(info['offdiag_cos']):.5f}", flush=True)
            else:
                print(f"[step {step}] loss={float(info['loss']):.4f}", flush=True)
        if step % FLAGS.eval_freq == 0:
            vals = []
            for _ in range(FLAGS.val_batches):
                vb = next(val_iter)
                if FLAGS.infonce_w > 0:
                    vb.pop("task_wrong1"); vb.pop("task_wrong2")
                vi = jax.device_get(eval_step(train_state, vb, None))
                vals.append(float(vi["loss"]))
            print(f"[step {step}] VAL_loss={sum(vals)/len(vals):.4f}", flush=True)
            train_state.model.save_pretrained(step=step, checkpoint_path=FLAGS.save_dir)

if __name__ == "__main__":
    app.run(main)
