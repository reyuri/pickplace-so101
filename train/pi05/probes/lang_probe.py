#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""决定性实验：pi05 到底读不读语言？

用户症状：桌上摆 3 个玩具，指令指定抓某一个，模型却按"位置"乱抓 —— 典型「动作捷径」。

本实验把「语言」与「画面」拆开：
  - 同一帧，换 3 个任务字符串，各重复 R 次
  - 由于 pi05 推理**不固定种子**（见 memory pi05-inference-unseeded），
    必须同时量出「同帧同指令重复跑」的噪声地板，否则任何 Δ 都不可解释。

判读：
  D_lang / D_noise ~ 1   -> 语言字符串对动作没有超出采样噪声的影响 = 语言条件实际失效
                            => 模型只能靠视觉选目标(最近/最前) => 正是"捷径"
  D_lang / D_noise >> 1  -> 语言确实被读进去了，问题在别处（映射错 / 场景分布）

另给 D_frame（换一帧、同指令）作为"输入真的变了"的尺度参照。
"""
import base64
import glob
import json
import os

import cv2
import numpy as np
import pandas as pd
import requests
from lerobot.datasets.lerobot_dataset import LeRobotDataset

DATA = "/root/autodl-tmp/lerobot_rey_toy_all"
URL = os.environ.get("PI05_URL", "http://127.0.0.1:6030")   # 换端口即可对比不同 ckpt
TAG = os.environ.get("PI05_TAG", "6030")
TASKS = [
    "Pick up the elephant and place it into the box",
    "Pick up the tree and place it into the box",
    "Pick up the ball and place it into the box",
]
R = 4                      # 每格重复次数（噪声地板用）
NAMES = ["j1", "j2", "j3", "j4", "j5", "grip"]


def img_u8(v):
    a = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
    if a.ndim == 3 and a.shape[0] == 3:
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        a = (np.clip(a, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return np.ascontiguousarray(a)


def b64j(hwc_rgb):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(hwc_rgb, cv2.COLOR_RGB2BGR))
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


def infer(ds, i, task):
    it = ds[i]
    payload = {
        "side": b64j(img_u8(it["observation.images.side"])),
        "eye_in_hand": b64j(img_u8(it["observation.images.eye_in_hand"])),
        "state": [float(x) for x in np.asarray(it["observation.state"], np.float32).reshape(-1)],
        "task": task,
    }
    d = requests.post(URL + "/infer", json=payload, timeout=180).json()
    return np.asarray(d["action"], np.float32), it


def main():
    files = sorted(glob.glob(f"{DATA}/data/**/*.parquet", recursive=True))
    parts = []
    for f in files:
        parts.append(pd.read_parquet(f))
    df = pd.concat(parts, ignore_index=True)
    print(f"[probe] parquet 列 = {list(df.columns)}")
    print(f"[probe] 总帧数 = {len(df)}")

    S = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    if "episode_index" not in df.columns:
        raise SystemExit("缺少 episode_index，无法定位 episode")
    ep = df["episode_index"].to_numpy()
    if "task_index" in df.columns:
        tk = df["task_index"].to_numpy()
    else:
        tk = np.zeros(len(df), np.int64)

    # 每个 task_index 取一条 episode，取该 episode 40% 处的一帧
    picks = []
    for t in sorted(set(tk.tolist())):
        m = tk == t
        eps = sorted(set(ep[m].tolist()))
        e = eps[len(eps) // 2]                       # 该任务的中间那条 episode
        idxs = np.where(ep == e)[0]
        picks.append(int(idxs[int(len(idxs) * 0.4)]))
    print(f"[probe] 每任务取一帧 -> 全局索引 {picks}  (task_index={sorted(set(tk.tolist()))})")

    ds = LeRobotDataset("reyqiao/lerobot_rey_toy_all", root=DATA)
    for i in picks:
        st = np.asarray(ds[i]["observation.state"], np.float32).reshape(-1)
        if not np.allclose(st, S[i], atol=1e-3):
            print(f"[probe] !! 索引错位 ds[{i}]={np.round(st,2).tolist()} vs parquet {np.round(S[i],2).tolist()}")
            raise SystemExit(2)

    print(f"[probe] serve = {requests.get(URL + '/', timeout=10).json()}")

    # A[frame][task][repeat] = 50x6 chunk
    A = {}
    for fi, i in enumerate(picks):
        for ti, t in enumerate(TASKS):
            for r in range(R):
                a, it = infer(ds, i, t)
                A[(fi, ti, r)] = a
                tail = a[-10:].mean(0)
                print(f"  f{fi} idx{i} task{ti} r{r}: state={np.round(np.asarray(it['observation.state'],np.float32).reshape(-1),1).tolist()}"
                      f" chunk末10步均={np.round(tail,1).tolist()}")

    def vec(a):        # 用 chunk 末 10 步均值代表"这一段要去哪"
        return a[-10:].mean(0)

    def pdiff(u, v):
        return np.abs(u - v)

    print("\n[probe] ================= 结果 =================")
    D_noise_all, D_lang_all, D_frame_all = [], [], []
    for fi in range(len(picks)):
        # 噪声地板：同帧同指令，重复之间的平均两两差
        dn = []
        for ti in range(len(TASKS)):
            for r1 in range(R):
                for r2 in range(r1 + 1, R):
                    dn.append(pdiff(vec(A[(fi, ti, r1)]), vec(A[(fi, ti, r2)])))
        dn = np.mean(dn, 0)
        # 语言差：同帧，不同指令的"重复配对均值"之间的平均两两差
        dm = {ti: np.mean([vec(A[(fi, ti, r)]) for r in range(R)], 0) for ti in range(len(TASKS))}
        dl = []
        for t1 in range(len(TASKS)):
            for t2 in range(t1 + 1, len(TASKS)):
                dl.append(pdiff(dm[t1], dm[t2]))
        dl = np.mean(dl, 0)
        D_noise_all.append(dn); D_lang_all.append(dl)
        print(f"\n  帧 f{fi} (idx {picks[fi]}):")
        print(f"    噪声地板 D_noise = {np.round(dn,2).tolist()}")
        print(f"    语言差   D_lang  = {np.round(dl,2).tolist()}")
        print(f"    比值     D_lang/D_noise = {np.round(dl / np.maximum(dn,1e-6),2).tolist()}")
        for ti, t in enumerate(TASKS):
            print(f"      指令 {t[9:].split(' and')[0]:8s} -> chunk末10步 = {np.round(dm[ti],1).tolist()}")

    # 换帧尺度：同指令，不同帧之间的差
    ref = TASKS[0]
    for fi in range(len(picks)):
        for fj in range(fi + 1, len(picks)):
            u = np.mean([vec(A[(fi, 0, r)]) for r in range(R)], 0)
            v = np.mean([vec(A[(fj, 0, r)]) for r in range(R)], 0)
            D_frame_all.append(pdiff(u, v))
    Df = np.mean(D_frame_all, 0) if D_frame_all else np.zeros(6)

    Dn = np.mean(D_noise_all, 0); Dl = np.mean(D_lang_all, 0)
    print(f"\n  ===== 汇总（{len(picks)} 帧 × {len(TASKS)} 指令 × {R} 次）=====")
    print(f"  关节      : {'  '.join(f'{n:>7s}' for n in NAMES)}")
    print(f"  噪声地板  : {'  '.join(f'{x:7.2f}' for x in Dn)}")
    print(f"  语言差    : {'  '.join(f'{x:7.2f}' for x in Dl)}")
    print(f"  换帧尺度  : {'  '.join(f'{x:7.2f}' for x in Df)}")
    print(f"  语言/噪声 : {'  '.join(f'{x:7.2f}' for x in Dl / np.maximum(Dn, 1e-6))}")
    print(f"  语言/换帧 : {'  '.join(f'{x:7.2f}' for x in Dl / np.maximum(Df, 1e-6))}")


if __name__ == "__main__":
    main()
