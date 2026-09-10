# -*- coding: utf-8 -*-
"""抽取 YOLO 标注用的 side 相机帧(沿时间轴采样),按 toy 分 train/val。

只抽 side(主视角)帧;每 toy 抽若干 episode、每个 episode 每隔 step 帧抽一张,
覆盖「玩具在桌/臂伸去/抓/举/放盒/回home」全程。标签类别(全项目统一 5 类):
  0 elephant, 1 tree, 2 ball, 3 box, 4 arm
每帧里实际只会出现该 toy(1个) + box + arm。

输出:
  annotate/side/train/<toy>/<split>_<ep>_<f>.jpg
  annotate/side/val/<toy>/<split>_<ep>_<f>.jpg
  annotate/side/manifest.csv   (filename, toy, src_ep, src_frame, split)
"""
import csv
import os
from pathlib import Path

import cv2

ROOT = Path(r"D:/My_Work/202510-findAremote/Projects/VLA/VLA_Yolo")
DATA = ROOT / "data" / "reyqiao"
OUT = ROOT / "annotate" / "side"

TOYS = ["elephant", "tree", "ball"]
# 每个 toy 采样的 episode(全局索引 0..44)
TRAIN_EPS = [0, 6, 12, 18, 24, 30]
VAL_EPS = [36, 42]
FRAME_STEP = 20          # 每 episode 每隔 20 帧抽 1 张(约 14 张/ep)

# 每 toy 在合并数据集里的偏移(不用,这里直接按 toy 独立目录读)
os.makedirs(OUT / "train", exist_ok=True)
os.makedirs(OUT / "val", exist_ok=True)

rows = []
for toy in TOYS:
    vid_dir = DATA / f"lerobot_rey_toy_{toy}" / "videos" / "observation.images.side" / "chunk-000"
    for split, eps in (("train", TRAIN_EPS), ("val", VAL_EPS)):
        out_dir = OUT / split / toy
        os.makedirs(out_dir, exist_ok=True)
        for ep in eps:
            vp = vid_dir / f"file-{ep:03d}.mp4"
            if not vp.exists():
                print(f"[warn] 缺视频 {vp}", flush=True)
                continue
            cap = cv2.VideoCapture(str(vp))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for f in range(0, n, FRAME_STEP):
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ok, fr = cap.read()
                if not ok or fr is None:
                    continue
                name = f"{split}_{ep}_{f:04d}.jpg"
                cv2.imwrite(str(out_dir / name), fr)
                rows.append({"filename": f"{toy}/{name}", "toy": toy,
                             "src_ep": ep, "src_frame": f, "split": split})
            cap.release()
        print(f"[{toy}] {split}: {len([x for x in rows if x['toy']==toy and x['split']==split])} frames", flush=True)

# manifest
with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["filename", "toy", "src_ep", "src_frame", "split"])
    w.writeheader()
    w.writerows(rows)

train_n = len([x for x in rows if x["split"] == "train"])
val_n = len([x for x in rows if x["split"] == "val"])
print(f"\nDONE: 共 {len(rows)} 帧  (train {train_n} + val {val_n})")
print(f"  目录: {OUT}")
