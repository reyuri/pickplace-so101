import os, sys, functools
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TFDS_DATA_DIR", "/root/autodl-tmp/octo_tfds_v2")
sys.path.insert(0, "/root/autodl-tmp")

import tensorflow as tf

# ---- KILL all GCS network probes. Server has no (working) access to googleapis,
# ---- and TFDS 4.x pings gs://tfds-data during Builder.__init__/info -> hangs 61s/retry.
# ---- Force: any gs:// path "does not exist" / dataset not on GCS so we build purely locally.
_orig_exists = tf.io.gfile.exists

def _safe_exists(path):
    p = os.fspath(path)
    if str(p).startswith("gs://"):
        return False            # remote dataset not present -> build fresh from SourceBuilder
    try:
        return _orig_exists(p)
    except Exception:
        return False

tf.io.gfile.exists = _safe_exists

import tensorflow_datasets as tfds
import tensorflow_datasets.core.utils.gcs_utils as _gcs
_gcs.is_dataset_on_gcs = staticmethod(lambda name: False)
_gcs.gcs_dataset_info_path = staticmethod(lambda name: None)
_gcs.gcs_listdir = staticmethod(lambda dir_name: None)
_gcs.download_gcs_dataset = staticmethod(lambda *a, **k: None)
_gcs._raise_gcs_endpoints_error = staticmethod(lambda: None)
from tensorflow_datasets.core.download import DownloadConfig

from lerobot_toy_builder.lerobot_toy_dataset import LerobotToy

# TFDS 4.9.2 + new etils/epath: `code_path` iterates module chain and hits a
# MultiplexedPath (no `.parts`) -> AttributeError. This path is only used to
# locate a checksums.tsv / url_infos file. Our builder downloads nothing, so
# short-circuit the whole triad to safe values.
import pathlib
LerobotToy.code_path = classmethod(lambda cls: pathlib.Path("/root/autodl-tmp/lerobot_toy_builder/lerobot_toy_dataset.py"))
LerobotToy._checksums_path = classmethod(lambda cls: None)
# DownloadManager.__init__ does `self._url_infos.update(url_infos)` -> needs a DICT,
# so expose `url_infos` as an instance attribute (not a callable classmethod).
LerobotToy.url_infos = property(lambda self: {})

NEW_DIR = "/root/autodl-tmp/octo_tfds_v2"
DC = DownloadConfig(try_download_gcs=False, register_checksums=False)

print(f"[build] LerobotToy(data_dir={NEW_DIR}) download_and_prepare (GCS masked) ...", flush=True)
b = LerobotToy(data_dir=NEW_DIR)
print("[construct] ok, not stuck on GCS", flush=True)
b.download_and_prepare(download_config=DC)          # no GCS -> build locally
print("[build] done", flush=True)

print("\n===== verify language_instruction is per-toy CLEAN =====", flush=True)
for split in ["train", "val"]:
    ds = tfds.load("lerobot_toy", split=split, data_dir=NEW_DIR)
    seen = {}
    for ex in ds.take(400):
        steps = ex["steps"]
        toy_id = ex["episode_metadata"]["toy"]
        lang = steps["language_instruction"].numpy().tolist()[:1]
        seen.setdefault(str(toy_id), set()).update(str(l) for l in lang)
    print(f"[{split}] per-toy instructions:")
    for toy, lset in seen.items():
        print(f"    {toy}: {sorted(lset)}")
print("[verify] done", flush=True)
