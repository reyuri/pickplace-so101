#!/usr/bin/env python
"""LeRobot v3.0 dataset  ->  Octo 训练用的逐 episode npz。

Octo 的 TFDS builder (``data/lerobot_toy_dataset.py``) 期望的输入是一堆
``<toy>/ep_<N>.npz``，每个 npz 是一条完整轨迹，键为：

    side    (T, S, S, 3) uint8    侧视相机（Brio 90）
    wrist   (T, S, S, 3) uint8    手眼相机
    action  (T, 6)       float32  6 维关节绝对位置
    state   (T, 6)       float32
    lang    ()           str      本条轨迹的语言指令

``ep_<N>.npz`` 里的 N 是**该 toy 数据集内**的 ``episode_index``（0..N-1）。
多个 toy 各自成目录，拼成一个全局数据集时由 builder 按 ``OFF`` 偏移。

为什么不用 LeRobotDataset 读：那条路要 torchcodec（编解码后端在
Windows/多数服务器上都装不出来）。这里直接 pyarrow 读 parquet + PyAV 解码 mp4，
只需要 ``pyarrow`` 和 ``av`` 两个纯 pip 依赖。

用法::

    python lerobot_to_octo_frames.py \
        --root /path/to/lerobot_rey_toy_elephant --toy elephant \
        --out  /path/to/octo_frames --size 256

    # 三个 toy 一次跑完（合并成 135 条的那个数据集也适用，--toy 只影响输出目录名）
    for t in ball elephant tree; do
      python lerobot_to_octo_frames.py --root lerobot_rey_toy_$t --toy $t --out octo_frames
    done

    # 跑完自查：逐条比对 npz 的 T 与 meta 里的 episode length
    python lerobot_to_octo_frames.py --root ... --toy ... --out ... --verify-only
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# 读 parquet 元数据
# --------------------------------------------------------------------------


def read_tasks(ds_root: Path) -> dict[int, str]:
    """meta/tasks.parquet -> {task_index: 指令文本}。"""
    import pyarrow.parquet as pq

    tbl = pq.read_table(ds_root / "meta" / "tasks.parquet")
    names = tbl.schema.names
    # v3.0 里指令文本存在索引列（名为空串或 __index_level_0__），task_index 是普通列
    idx_col = "task_index"
    text_col = next(c for c in names if c != idx_col)
    d = tbl.to_pydict()
    return {int(i): str(t) for i, t in zip(d[idx_col], d[text_col])}


def read_episodes(ds_root: Path) -> list[dict]:
    """meta/episodes/chunk-*/*.parquet -> 每条的 length / 各相机的视频分段。"""
    import pyarrow.parquet as pq

    files = sorted(glob.glob(str(ds_root / "meta" / "episodes" / "chunk-*" / "*.parquet")))
    if not files:
        raise SystemExit(f"找不到 meta/episodes/*.parquet: {ds_root}")

    out: list[dict] = []
    for f in files:
        d = pq.read_table(f).to_pydict()
        n = len(d["episode_index"])
        for i in range(n):
            ep = int(d["episode_index"][i])
            videos = {}
            for col in d:
                # videos/<cam>/chunk_index 等四条同前缀，收成一个 dict
                if not col.startswith("videos/"):
                    continue
                _, cam, field = col.split("/", 2)
                videos.setdefault(cam, {})[field] = d[col][i]
            out.append(
                {
                    "episode_index": ep,
                    "length": int(d["length"][i]),
                    "from_index": int(d["dataset_from_index"][i]),
                    "to_index": int(d["dataset_to_index"][i]),
                    "videos": videos,
                    "tasks": d["tasks"][i],
                }
            )
    out.sort(key=lambda e: e["episode_index"])
    return out


def read_frames(ds_root: Path) -> dict[str, np.ndarray]:
    """data/chunk-*/*.parquet -> {列名: (total_frames, D) ndarray}，按全局 index 排好序。

    只读动作/状态这类小列；视频不在这里。
    """
    import pyarrow.parquet as pq

    files = sorted(glob.glob(str(ds_root / "data" / "chunk-*" / "*.parquet")))
    if not files:
        raise SystemExit(f"找不到 data/*.parquet: {ds_root}")

    want = ("action", "observation.state", "index", "episode_index", "frame_index")
    chunks: dict[str, list[np.ndarray]] = {k: [] for k in want}
    for f in files:
        d = pq.read_table(f, columns=list(want)).to_pydict()
        if not d["index"]:
            continue  # 空分片
        for k in want:
            # 每格是 list/标量，统一拍平成 (N, D)
            chunks[k].append(np.stack([np.asarray(v, dtype=np.float64).reshape(-1) for v in d[k]]))

    cat = {k: np.concatenate(v, axis=0) for k, v in chunks.items()}
    # index 是标量列，拍平后再排；argsort 默认按最后一维排，直接喂 (N,1) 会得到错的结果
    order = np.argsort(cat["index"].ravel(), kind="stable")
    return {k: v[order] for k, v in cat.items()}


# --------------------------------------------------------------------------
# 视频解码
# --------------------------------------------------------------------------


def video_path(ds_root: Path, cam: str, chunk: int, file: int) -> Path:
    return ds_root / "videos" / cam / f"chunk-{chunk:03d}" / f"file-{file:03d}.mp4"


def decode_segment(path: Path, from_ts: float, n: int, size: int, to_ts: float | None = None) -> np.ndarray:
    """按时间戳定位，解出 n 帧并缩放到 (size, size)，返回 (n, size, size, 3) uint8 RGB。

    from_ts/to_ts 来自 meta/episodes 的 from_timestamp/to_timestamp；本数据集
    一条 mp4 就是一条 episode（from_ts=0），但按时间戳切更稳，能兼容多条合并进
    同一 mp4 的情况（aggregate_datasets 会这样拼）。
    """
    import av

    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        # 跳到起始时间戳
        t0 = stream.start_time if from_ts <= 0 else int(from_ts / float(stream.time_base))
        container.seek(t0, stream=stream, backward=True)
        for frame in container.decode(stream):
            ts = float(frame.pts * stream.time_base) if frame.pts is not None else 0.0
            if ts + 1e-6 < from_ts:
                continue
            if to_ts is not None and ts > to_ts + 1e-6:
                break
            frames.append(resize(frame.to_ndarray(format="rgb24"), size))
            if len(frames) == n:
                break

    if len(frames) != n:
        raise RuntimeError(f"{path.name}: 只解出 {len(frames)} 帧，期望 {n} 帧")
    return np.stack(frames)


def resize(img: np.ndarray, size: int) -> np.ndarray:
    """缩放到 size x size。有 cv2 用 cv2（和训练侧一致），否则退回 PIL。"""
    if img.shape[0] == size and img.shape[1] == size:
        return img
    try:
        import cv2

        return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    except ImportError:
        from PIL import Image

        return np.asarray(Image.fromarray(img).resize((size, size), Image.BILINEAR))


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path, help="LeRobot v3.0 数据集根目录（含 meta/ data/ videos/）")
    ap.add_argument("--toy", required=True, help="输出子目录名，同时也是轨迹的语言指令来源标签")
    ap.add_argument("--out", required=True, type=Path, help="octo_frames 输出根目录")
    ap.add_argument("--size", type=int, default=256, help="side 相机缩放边长（默认 256）")
    ap.add_argument("--wrist-size", type=int, default=None, help="手眼相机缩放边长（默认同 --size）")
    ap.add_argument("--side-key", default=None, help="侧视相机的 feature key（默认自动取非 eye_in_hand 的那个）")
    ap.add_argument("--wrist-key", default="observation.images.eye_in_hand", help="手眼相机的 feature key")
    ap.add_argument("--only", type=int, nargs="*", default=None, help="只处理这些 episode_index（调试用）")
    ap.add_argument("--verify-only", action="store_true", help="不写文件，只比对已有 npz 的 T 与 meta length")
    ap.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的 npz")
    args = ap.parse_args()

    root: Path = args.root
    if not (root / "meta" / "info.json").is_file():
        raise SystemExit(f"不是 LeRobot 数据集根目录: {root}")

    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]

    wrist_key = args.wrist_key
    side_key = args.side_key or next((k for k in video_keys if k != wrist_key), None)
    if side_key is None:
        raise SystemExit(f"推不出侧视相机 key，请用 --side-key 指定（可用: {video_keys}）")
    wrist_size = args.wrist_size or args.size

    print(f"[{args.toy}] {root}")
    print(f"  版本 {info['codebase_version']} / {info['total_episodes']} 条 / {info['total_frames']} 帧 / {info['fps']} fps")
    print(f"  相机 side={side_key} -> {args.size}px, wrist={wrist_key} -> {wrist_size}px")

    episodes = read_episodes(root)
    tasks = read_tasks(root)
    out_dir = args.out / args.toy
    out_dir.mkdir(parents=True, exist_ok=True)

    # --verify-only: 只比对长度
    if args.verify_only:
        bad = 0
        for ep in episodes:
            p = out_dir / f"ep_{ep['episode_index']}.npz"
            if not p.is_file():
                print(f"  缺失 {p.name}")
                bad += 1
                continue
            with np.load(p) as z:
                got = z["side"].shape[0]
            if got != ep["length"]:
                print(f"  {p.name}: T={got} 期望 {ep['length']}")
                bad += 1
        print(f"  自查: {len(episodes) - bad}/{len(episodes)} 条长度一致")
        return 1 if bad else 0

    frames = read_frames(root)
    idx_of = {int(e): i for i, e in enumerate(frames["index"].ravel())}

    done = skipped = 0
    for ep in episodes:
        n = ep["episode_index"]
        if args.only is not None and n not in args.only:
            continue
        dst = out_dir / f"ep_{n}.npz"
        if dst.is_file() and not args.overwrite:
            print(f"  跳过 ep_{n}（已存在，--overwrite 可覆盖）")
            skipped += 1
            continue

        # 动作/状态：meta 里的 [from_index, to_index) 就是这段在全局 index 上的范围
        lo, hi = idx_of[ep["from_index"]], idx_of[ep["to_index"] - 1] + 1
        action = frames["action"][lo:hi].astype(np.float32)
        state = frames["observation.state"][lo:hi].astype(np.float32)
        if len(action) != ep["length"]:
            raise RuntimeError(f"ep_{n}: parquet 给了 {len(action)} 帧，meta length={ep['length']}")

        v = ep["videos"][side_key]
        side = decode_segment(
            video_path(root, side_key, int(v["chunk_index"]), int(v["file_index"])),
            float(v["from_timestamp"]), ep["length"], args.size, float(v["to_timestamp"]),
        )
        v = ep["videos"][wrist_key]
        wrist = decode_segment(
            video_path(root, wrist_key, int(v["chunk_index"]), int(v["file_index"])),
            float(v["from_timestamp"]), ep["length"], wrist_size, float(v["to_timestamp"]),
        )

        # 指令：优先用 meta 里逐条记的 tasks，退回 tasks.parquet
        lang = ep["tasks"]
        if isinstance(lang, (list, tuple, np.ndarray)):
            lang = str(lang[0]) if len(lang) else ""
        if not lang:
            lang = tasks.get(int(frames["task_index"][lo, 0]), "")

        np.savez_compressed(
            dst,
            side=side,
            wrist=wrist,
            action=action,
            state=state,
            lang=np.array(str(lang)),
        )
        done += 1
        if done % 10 == 0 or done == 1:
            print(f"  [{done}] ep_{n} T={ep['length']} lang={lang!r}")

    print(f"  完成: 写入 {done} 条, 跳过 {skipped} 条 -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
