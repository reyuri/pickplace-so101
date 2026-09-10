# -*- coding: utf-8 -*-
"""Step-1: 为 A2 离线生成"目标玩具"检测框 sidecar(仅目标类 + 最高置信 1 框)。

规则(用户 2026-09-05 定):训练图只画【目标 toy】一个粗红框,其他任何检测框都不画;
  无目标框的帧 -> 该帧 box=null(注入红框为空,保留帧)。

输出 /root/autodl-tmp/yolo_target_boxes/:
  - ep_{ep:03d}.json       每 ep:  {frame_index: [cls,cx,cy,w,h,conf](归一化) | null}
  - stats.json             汇总: 总帧/有目标/无目标/每toy/置信度分布
  - only_redbox_montage.jpg 抽样"仅红框"拼图(供人工确认)

跑法(服务器): /root/autodl-tmp/venv_yolo/bin/python /root/autodl-tmp/gen_yolo_targetbox_sidecar.py
"""
import glob
import json
import os

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

DS = "/root/autodl-tmp/lerobot_rey_toy_all"
BEST = "/root/autodl-tmp/yolo_runs/yolo11n_toy5-3/weights/best.pt"
OUTDIR = "/root/autodl-tmp/yolo_target_boxes"
SESSION_DIR = os.path.dirname(BEST)
VIDEO = DS + "/videos/observation.images.side/chunk-000"
IMGSZ = [480, 640]          # (h, w) 保持 640x480 源图宽高比
CONF_MIN = 0.1              # 尽量多抓到目标,置信度再决定是否保留
DO_MONTAGE = True

os.makedirs(OUTDIR, exist_ok=True)
model = YOLO(BEST)

# ---- 1. task_index -> toy -> yolo 类 id ----
tasks = pd.read_parquet(DS + "/meta/tasks.parquet")
task_str_by_idx = tasks.index.tolist()  # tasks.parquet 的行: task_string -> task_index
# tasks.parquet 结构: 列 'task_index', 索引是 task_string(见早前读档)
TOY2YOLO = {"elephant": 0, "tree": 1, "ball": 2}
task_toy = {}


def toy_from_task(s):
    for t in ("elephant", "tree", "ball"):
        if t in s:
            return t
    return None


for i, task_str in enumerate(task_str_by_idx):
    task_toy[i] = toy_from_task(task_str)
print("task_index->toy:", task_toy, flush=True)

# ---- 2. ep -> task_index(读帧级 data parquet,只取两列) ----
frames_task = []
for fp in sorted(glob.glob(os.path.join(DS, "data", "chunk-000", "file-*.parquet"))):
    d = pd.read_parquet(fp, columns=["episode_index", "task_index"])
    frames_task.append(d)
df = pd.concat(frames_task, ignore_index=True)
ep_task = df.drop_duplicates(subset=["episode_index"], keep="last") \
            .set_index("episode_index")["task_index"].to_dict()
print("ep->task 覆盖:", len(ep_task), flush=True)

# ---- 3. ep meta: 顺序 + 帧长 ----
ep_meta = pd.read_parquet(DS + "/meta/episodes/chunk-000/file-000.parquet",
                          columns=["episode_index", "length"]) \
            .sort_values("episode_index")
episodes = ep_meta["episode_index"].tolist()

# ---- 4. 逐 ep 解码 + YOLO + 只取目标类最高置信 ----
stats = {"total_frames": 0, "with_target": 0, "no_target": 0,
         "per_toy": {}, "conf_buckets": {"<0.3": 0, "0.3-0.5": 0, "0.5-0.7": 0, "0.7-0.9": 0, ">=0.9": 0}}


def detect_batch(frames, tgt_id, out_map, base_fidx):
    """对一批 BGR 帧跑 YOLO,只记录目标类置信度最高的框。"""
    if not frames:
        return
    res = model.predict(source=frames, imgsz=IMGSZ, conf=CONF_MIN, verbose=False)
    for j, r in enumerate(res):
        fidx = base_fidx + j
        if len(r.boxes) == 0:
            out_map[fidx] = None
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().int().tolist()
        idxs = [i for i, c in enumerate(cls) if c == tgt_id]
        if not idxs:
            out_map[fidx] = None
            continue
        k = max(idxs, key=lambda i: conf[i])
        x1, y1, x2, y2 = xyxy[k]
        w, h = IMGSZ[1], IMGSZ[0]
        box = [int(tgt_id), (x1 + x2) / 2 / w, (y1 + y2) / 2 / h,
               (x2 - x1) / w, (y2 - y1) / h, float(conf[k])]
        out_map[fidx] = box


mc = 0
for ep in episodes:
    tgt_toy = task_toy.get(ep_task.get(ep), None)
    if tgt_toy is None:
        print(f"[warn] ep{ep} 无 task/toy, 跳过", flush=True)
        continue
    tgt_id = TOY2YOLO[tgt_toy]
    vp = f"{VIDEO}/file-{ep:03d}.mp4"
    cap = cv2.VideoCapture(vp)
    out_map = {}
    frames, base = [], None
    fi = 0
    while True:
        ret, fr = cap.read()
        if not ret or fr is None:
            break
        if base is None:
            base = fi
        frames.append(fr)
        if len(frames) >= 48:
            detect_batch(frames, tgt_id, out_map, base)
            frames = []
            base = None
        fi += 1
    cap.release()
    if frames:
        detect_batch(frames, tgt_id, out_map, base)

    # 统计
    n = len(out_map)
    stats["total_frames"] += n
    toy_key = tgt_toy
    stats["per_toy"].setdefault(toy_key, {"frames": 0, "with": 0, "no": 0})
    stats["per_toy"][toy_key]["frames"] += n
    for f, b in out_map.items():
        if b is None:
            stats["no_target"] += 1
            stats["per_toy"][toy_key]["no"] += 1
        else:
            stats["with_target"] += 1
            stats["per_toy"][toy_key]["with"] += 1
            c = b[5]
            if c < 0.3:
                stats["conf_buckets"]["<0.3"] += 1
            elif c < 0.5:
                stats["conf_buckets"]["0.3-0.5"] += 1
            elif c < 0.7:
                stats["conf_buckets"]["0.5-0.7"] += 1
            elif c < 0.9:
                stats["conf_buckets"]["0.7-0.9"] += 1
            else:
                stats["conf_buckets"][">=0.9"] += 1
    # 存 sidecar(把 int key 序列化为 str)
    with open(f"{OUTDIR}/ep_{ep:03d}.json", "w") as fh:
        json.dump({str(k): (v if v is None else [float(x) for x in v]) for k, v in out_map.items()}, fh)
    mc += 1
    if mc % 15 == 0 or ep == episodes[-1]:
        print(f"进度 {mc}/{len(episodes)} ep{ep} toy={tgt_toy} frames={n} "
              f"no_target={(stats['no_target'])}", flush=True)

with open(f"{OUTDIR}/stats.json", "w") as fh:
    json.dump(stats, fh, indent=2)
print("\n===== STATS =====")
print("总帧", stats["total_frames"], "有目标", stats["with_target"],
      "无目标", stats["no_target"],
      f"检出率 {stats['with_target']/max(stats['total_frames'],1)*100:.1f}%")
print("per_toy:", json.dumps(stats["per_toy"], indent=2, ensure_ascii=False))
print("conf_buckets:", stats["conf_buckets"])

# ---- 5. "仅红框" 抽查拼图(只画目标粗红框,无其他线) ----
if DO_MONTAGE:
    RED = (0, 0, 255)
    import random
    random.seed(7)
    tiles = []
    # 跨 3 toy, 每 toy 抽 4 帧(有框/无框都行)
    for toy, yid in TOY2YOLO.items():
        eps_for_toy = [ep for ep in episodes if task_toy.get(ep_task.get(ep)) == toy]
        if not eps_for_toy:
            continue
        for _ in range(4):
            ep = random.choice(eps_for_toy)
            data = json.load(open(f"{OUTDIR}/ep_{ep:03d}.json"))
            # 均匀抽一个帧
            ks = sorted(int(k) for k in data)
            if not ks:
                continue
            fidx = random.choice(ks)
            b = data.get(str(fidx))
            cap = cv2.VideoCapture(f"{VIDEO}/file-{ep:03d}.mp4")
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, fr = cap.read()
            cap.release()
            if not ok or fr is None:
                continue
            if b is not None:
                _, cx, cy, bw, bh, conf = b
                w, h = IMGSZ[1], IMGSZ[0]
                x1 = int((cx - bw / 2) * w); y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w); y2 = int((cy + bh / 2) * h)
                cv2.rectangle(fr, (x1, y1), (x2, y2), RED, 3)
                cv2.putText(fr, f"{toy} {conf:.2f}", (x1, max(y1 - 6, 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)
            ann = cv2.resize(fr, (426, 320))
            tag = f"ep{ep} f{fidx}" + (" | NO-BOX" if b is None else "")
            cv2.rectangle(ann, (0, 0), (425, 22), (0, 0, 0), -1)
            cv2.putText(ann, tag, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            tiles.append(ann)
    if tiles:
        cols = 4
        rows = (len(tiles) + cols - 1) // cols
        canvas = np.zeros((rows * 320, cols * 426, 3), np.uint8)
        for i, t in enumerate(tiles):
            rr, cc = divmod(i, cols)
            canvas[rr * 320:(rr + 1) * 320, cc * 426:(cc + 1) * 426] = t
        mp = f"{OUTDIR}/only_redbox_montage.jpg"
        cv2.imwrite(mp, canvas)
        print("抽样拼图:", mp, f"({len(tiles)} tiles)", flush=True)
print("\nDONE")
