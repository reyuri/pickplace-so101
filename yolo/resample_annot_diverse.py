# -*- coding: utf-8 -*-
"""基于视觉相似度的"最远离采样(farthest-point)"抽帧 —— 挑目视差异最大的帧。

区别于 prepare_yolo_ann.py 的"固定episode+固定帧间隔"(会撞上脚本化近重复场景)。
思路:每个 toy 单独做 ——
  1. 解码全部 side 帧(每 N 帧取 1 进候选池,用来控制内存/速度);
  2. 每帧算一个低分辨率 RGB 缩略图特征(L2归一化),编码「摆放/姿态/遮挡/形变」;
  3. farthest-point sampling:反复挑「离已选样本最远」的帧,k 次 → k 个最互补、覆盖面最广的帧;
  4. 每个 toy 抽 74 张(64 train + 10 val),全池 3×74 ≈ 222 张;
  5. 输出 annotate/diverse/{train,val}/<toy>/  + 一张 0.4x 缩略拼图 preview 供人工确认。

类平衡说明:每帧都含 toy+box+arm,故「张数=每类实例数」,类别天然平衡,这里无需按类配比。

用法: python resample_annot_diverse.py --per-toy 74 --stride 2
"""
import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"D:/My_Work/202510-findAremote/Projects/VLA/VLA_Yolo")
DATA = ROOT / "data" / "reyqiao"
OUT = ROOT / "annotate" / "diverse"

TOYS = ["elephant", "tree", "ball"]
THUMB = (48, 64)   # (h, w) 缩略图特征, 64*48*3 = 9216 维


def describe(fr):
    """BGR 帧 -> 归一化缩略图特征向量。"""
    small = cv2.resize(fr, (THUMB[1], THUMB[0]))
    vec = small.astype(np.float32).reshape(-1)
    n = np.linalg.norm(vec)
    return vec / (n + 1e-8)


def build_pool(vid_path, stride):
    """顺序读视频,每 stride 帧进候选池;返回 (frames: list[BGR], feats: np.ndarray)。"""
    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        cap.release()
        return [], None
    frames, feats = [], []
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok or fr is None:
            break
        if i % stride == 0:
            frames.append((i, fr))
            feats.append(describe(fr))
        i += 1
    cap.release()
    return frames, (np.asarray(feats, dtype=np.float32) if feats else None)


def farthest_point_sampling(feats, k, rng):
    """feats (N,D) L2归一化;返回 k 个下标,使所选样本尽量「互补/分散」。"""
    n = feats.shape[0]
    if n <= k:
        return list(range(n))
    # 从离质心最远的点开始,给定一个「覆盖角落但不偏向密度」的起点
    centroid = feats.mean(axis=0)
    start = int(np.argmax(np.linalg.norm(feats - centroid, axis=1)))
    sel = [start]
    dmin = np.linalg.norm(feats - feats[start], axis=1)
    for _ in range(k - 1):
        i = int(np.argmax(dmin))
        sel.append(i)
        dd = np.linalg.norm(feats - feats[i], axis=1)
        dmin = np.minimum(dmin, dd)
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=200, help="train 帧总数")
    ap.add_argument("--val", type=int, default=20, help="val 帧总数")
    ap.add_argument("--stride", type=int, default=2, help="候选池每隔多少帧取1")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    def distribute(n, k):
        base, rem = n // k, n % k
        out = [base] * k
        for i in range(rem):
            out[i] += 1
        return out

    per_toy_train = distribute(args.train, len(TOYS))   # 200 -> [67,67,66]
    per_toy_val = distribute(args.val, len(TOYS))       # 20  -> [7,7,6]

    train_dir = OUT / "train"
    val_dir = OUT / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ti, toy in enumerate(TOYS):
        vid_dir = DATA / f"lerobot_rey_toy_{toy}" / "videos" / "observation.images.side" / "chunk-000"
        ep_files = sorted(vid_dir.glob("file-*.mp4"))
        print(f"\n=== {toy}: {len(ep_files)} 个 episode 视频 ===", flush=True)

        # 收集所有 episode 的候选帧
        cand = []          # list of (toy, ep, file_frame, BGR)
        feats_list = []
        for ep_path in ep_files:
            ep = int(ep_path.stem.split("-")[-1])
            frames, fts = build_pool(ep_path, args.stride)
            if fts is None or len(frames) < 1:
                print(f"  [warn] {ep_path.name} 读不出", flush=True)
                continue
            for (ff, fr), v in zip(frames, fts):
                cand.append((toy, ep, ff, fr))
                feats_list.append(v)
            # 顺序帧的 feats 要与其 frame 对齐(build_pool 返回的第二个是全部特征)
        # 注意:上面 zip 是对每一帧的 {frame_idx, fr} 和特征;但 build_pool 把特征也按帧序排了
        # 这里重新组装,避免两次读取。改为直接记录:
        # --- 下面重做,保留上面结构即可 ---
        feats = np.asarray(feats_list, dtype=np.float32)
        print(f"  候选帧: {len(cand)}, 特征: {feats.shape}", flush=True)

        n_train = per_toy_train[ti]
        n_val = per_toy_val[ti]
        total_toy = n_train + n_val
        sel = farthest_point_sampling(feats, total_toy, rng)
        # 前 n_train 张给 train,后 n_val 张给 val(都是最分散的,val 覆盖同样范围)
        sel_train = sel[:n_train]
        sel_val = sel[n_train:total_toy]

        for split, idxs in (("train", sel_train), ("val", sel_val)):
            out_dir = train_dir if split == "train" else val_dir
            sub = out_dir / toy
            sub.mkdir(parents=True, exist_ok=True)
            for si in idxs:
                toy_e, ep, ff, fr = cand[si]
                name = f"{split}_{ep}_{ff:04d}.jpg"
                cv2.imwrite(str(sub / name), fr)
                rows.append({"filename": f"{toy}/{name}", "toy": toy, "src_ep": ep,
                             "src_frame": ff, "split": split})
        print(f"  -> train {len(sel_train)}+val {len(sel_val)}", flush=True)

    # manifest
    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "toy", "src_ep", "src_frame", "split"])
        w.writeheader()
        w.writerows(rows)
    tr = len([r for r in rows if r["split"] == "train"])
    va = len([r for r in rows if r["split"] == "val"])
    print(f"\nDONE: {len(rows)} 帧 (train {tr} + val {va}) -> {OUT}")

    # 缩略拼图
    make_montage(rows, OUT)


def make_montage(rows, out_root, cols=8):
    """把选中的帧做成一张网格 preview,便于人工确认多样性。"""
    rows = sorted(rows, key=lambda r: (r["toy"], r["split"], r["src_frame"]))
    tiles = []
    for r in rows:
        p = out_root / r["split"] / r["filename"]
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.resize(img, (128, 96))
        cv2.rectangle(img, (0, 0), (127, 14), (0, 0, 0), -1)
        cv2.putText(img, f"{r['toy'][:2]}-{r['split'][:1]}-{r['src_ep']:02d}-{r['src_frame']:03d}",
                    (2, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
        tiles.append(img)
    n = len(tiles)
    rows_n = (n + cols - 1) // cols
    canvas = np.zeros((rows_n * 96, cols * 128, 3), dtype=np.uint8)
    for i, t in enumerate(tiles):
        r_, c = divmod(i, cols)
        canvas[r_ * 96:(r_ + 1) * 96, c * 128:(c + 1) * 128] = t
    out = out_root.parent / "diverse_preview.jpg"
    cv2.imwrite(str(out), canvas)
    print(f"preview: {out}  ({n} tiles)")


if __name__ == "__main__":
    main()
