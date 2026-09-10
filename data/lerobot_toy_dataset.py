import os, glob, json, numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

class LerobotToy(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.1")
    RELEASE_NOTES = {
        "1.0.0": "lerobot dual-cam toy to octo rlds",
        "1.0.1": "stratified 120/15 train-val split (val=ep%9==0, 5/toy) + proprio state scaled to [-2,2]",
    }

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Sequence({
                    "observation": tfds.features.FeaturesDict({
                        "side": tfds.features.Image(shape=(256,256,3), dtype=tf.uint8),
                        "eye_in_hand": tfds.features.Image(shape=(256,256,3), dtype=tf.uint8),
                        "state": tfds.features.Tensor(shape=(6,), dtype=tf.float32),
                    }),
                    "action": tfds.features.Tensor(shape=(6,), dtype=tf.float32),
                    "reward": tfds.features.Scalar(dtype=tf.float32),
                    "discount": tfds.features.Scalar(dtype=tf.float32),
                    "is_first": tfds.features.Scalar(dtype=tf.bool),
                    "is_last": tfds.features.Scalar(dtype=tf.bool),
                    "is_terminal": tfds.features.Scalar(dtype=tf.bool),
                    "language_instruction": tfds.features.Scalar(dtype=tf.string),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "toy": tfds.features.Scalar(dtype=tf.string),
                    "file_path": tfds.features.Scalar(dtype=tf.string),
                }),
                "len": tfds.features.Scalar(dtype=tf.int32),
            }),
        )

    def _split_generators(self, dl_manager):
        # two splits over the SAME frames; _generate_examples() decides per-episode
        # env-overridable so the SAME builder can also build the redbox variant
        frames_root = os.environ.get("OCTO_FRAMES_ROOT", "/root/autodl-tmp/octo_frames")
        return [
            tfds.core.SplitGenerator(
                name="train",
                gen_kwargs=dict(frames_root=frames_root, split="train")),
            tfds.core.SplitGenerator(
                name="val",
                gen_kwargs=dict(frames_root=frames_root, split="val")),
        ]

    def _generate_examples(self, frames_root, split):
        # state is M100 ([-100,100]); scale to [-2,2] to match Octo's proprio convention
        STATE_SCALE = 50.0
        # per-toy task instruction (fix: was constant "Pick up the tree..." -> model text-blind)
        # template is env-overridable so the redbox variant can use the A3-style prompt
        _template = os.environ.get("OCTO_LANG_TEMPLATE",
                                   "Pick up the {toy} and place it into the box")
        TOY_TASK = {t: _template.format(toy=t) for t in ("elephant", "tree", "ball")}
        n = 0
        for toy in sorted(os.listdir(frames_root)):
            toy_dir = os.path.join(frames_root, toy)
            if not os.path.isdir(toy_dir):
                continue
            for npz in sorted(glob.glob(os.path.join(toy_dir, "ep_*.npz"))):
                ep = n                     # global episode counter -> stable across both splits
                n += 1                     # count EVERY episode so ep%9 is identical per split
                is_val = (ep % 9 == 0)     # 5 per toy (ep 0,9,18,27,36 within each toy block of 45)
                if (split == "val") != is_val:
                    continue
                d = np.load(npz)
                side = d["side"]; wrist = d["wrist"]
                action = d["action"]; state = d["state"].astype(np.float32) / STATE_SCALE
                lang = TOY_TASK[toy]
                T = side.shape[0]
                yield f"{toy}_ep{ep}", {
                    "steps": {
                        "observation": {
                            "side": side.astype(np.uint8),
                            "eye_in_hand": wrist.astype(np.uint8),
                            "state": state.astype(np.float32),
                        },
                        "action": action.astype(np.float32),
                        "reward": np.zeros((T,), np.float32),
                        "discount": np.ones((T,), np.float32),
                        "is_first": np.array([True]+[False]*(T-1)),
                        "is_last": np.array([False]*(T-1)+[True]),
                        "is_terminal": np.zeros((T,), bool),
                        "language_instruction": np.array([lang]*T),
                    },
                    "episode_metadata": {"toy": toy, "file_path": npz},
                    "len": T,
                }
