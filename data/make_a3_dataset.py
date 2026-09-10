# -*- coding: utf-8 -*-
"""A3 数据集: 基于 A2 的 redbox 集, 只改 3 个 task 串为 box-aware 版。其余全硬链接(省磁盘)。

新 task 串(用户 2026-09-05 定): "Pick up the <toy> in red box and place it into the box"
   0: Pick up the elephant in red box and place it into the box
   1: Pick up the tree    in red box and place it into the box
   2: Pick up the ball    in red box and place it into the box

机制已验证: data 帧只有 task_index, task 字符串唯一来源 = meta/tasks.parquet, info.json 无字符串表。
跑法(服务器, smolvla 环境): python /root/autodl-tmp/make_a3_dataset.py
产物: /root/autodl-tmp/lerobot_rey_toy_all_redbox_a3
"""
import os
import shutil
import pandas as pd
from lerobot.datasets.lerobot_dataset import LeRobotDataset

SRC = "/root/autodl-tmp/lerobot_rey_toy_all_redbox"
DST = "/root/autodl-tmp/lerobot_rey_toy_all_redbox_a3"
TOYS = ["elephant", "tree", "ball"]


def prompt(s):
    return f"Pick up the {s} in red box and place it into the box"


# ---- 1. 硬链接拷贝 data / videos / meta(除 tasks.parquet) ----
os.makedirs(DST, exist_ok=True)

# data: 帧 parquet(27 个分组文件)
if not os.path.exists(os.path.join(DST, "data")):
    shutil.copytree(os.path.join(SRC, "data"), os.path.join(DST, "data"),
                    copy_function=os.link)
# videos: 重编码后的 side + eye_in_hand
if not os.path.exists(os.path.join(DST, "videos")):
    shutil.copytree(os.path.join(SRC, "videos"), os.path.join(DST, "videos"),
                    copy_function=os.link)
# meta: 除 tasks.parquet 外全部硬链接(episodes/ / info.json / stats.json)
metasrc = os.path.join(SRC, "meta")
metadst = os.path.join(DST, "meta")
os.makedirs(metadst, exist_ok=True)
for root, _, files in os.walk(metasrc):
    rel = os.path.relpath(root, metasrc)
    target = os.path.join(metadst, rel)
    os.makedirs(target, exist_ok=True)
    for f in files:
        if f == "tasks.parquet":
            continue
        sf = os.path.join(root, f)
        tf = os.path.join(target, f)
        if os.path.exists(tf):
            os.remove(tf)
        os.link(sf, tf)
print("[a3] 硬链接 data/videos/meta 完成", flush=True)

# ---- 2. 重写 tasks.parquet(唯一真正改的文件) ----
ot = pd.read_parquet(os.path.join(metasrc, "tasks.parquet"))
new_index = [prompt(TOYS[int(i)]) for i in ot["task_index"]]
ot.index = new_index  # 保持 task_index 0/1/2 不变, 只改字符串
ot.to_parquet(os.path.join(metadst, "tasks.parquet"))
print("[a3] tasks.parquet 重写为 box-aware 版:\n", ot.to_string(), flush=True)

# ---- 3. 端到端验证: LeRobotDataset 加载 + 取样本 task 串 ----
ds = LeRobotDataset(DST)
print(f"[a3] num_episodes={ds.num_episodes} num_frames={ds.num_frames}", flush=True)
samples = set()
for g in range(0, max(ds.num_frames, 1)):
    s = ds[g]
    samples.add(s["task"])
    if len(samples) == 3:
        break
print("[a3] 采样到的 task 串:", sorted(samples), flush=True)
print("[a3] A3_DATASET_DONE", flush=True)
