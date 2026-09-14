#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""直接回答用户的原始假设：「224 是不是太小了？」

做法：既然模型内部固定把图缩到 224×168 内容，那就在**喂进去之前**先把图降采样，
再让它放大到 224 —— 等价于「相机本身分辨率更低」。

判读：
  - 若降到 ~112 宽仍保持高相关 -> 模型用的是粗特征，224 **不是**瓶颈，加到 512 也不会救
  - 若一降就崩 -> 模型在用接近 224 极限的细节 -> 分辨率确实是限制项，提高会有收益

用上一版已经验证过的 t0 取帧（前检 r=+0.987，目标已定死）。
"""
import base64
import glob
import os

import cv2
import numpy as np
import pandas as pd
import requests
from lerobot.datasets.lerobot_dataset import LeRobotDataset

DATA = "/root/autodl-tmp/lerobot_rey_toy_all"
URL = os.environ.get("PI05_URL", "http://127.0.0.1:6030")
PER_TASK = int(os.environ.get("PI05_PER", "8"))
THRESH = 5.0
SCALES = [1.0, 0.50, 0.35, 0.25, 0.175, 0.125]
TASKS = ["elephant", "tree", "ball"]


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


def down(hwc, s):
    """模拟低分辨率相机：缩到 s 倍（面积插值），喂给 serve；模型自己会再缩到 224。"""
    if s >= 0.999:
        return hwc
    h, w = hwc.shape[:2]
    return cv2.resize(hwc, (max(8, int(round(w * s))), max(6, int(round(h * s)))),
                      interpolation=cv2.INTER_AREA)


def main():
    files = sorted(glob.glob(f"{DATA}/data/**/*.parquet", recursive=True))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    ep = df["episode_index"].to_numpy()
    tk = df["task_index"].to_numpy()
    A, L = {}, {}
    for e in sorted(set(ep.tolist())):
        m = ep == e
        A[int(e)] = np.stack(df.loc[m, "action"].to_numpy()).astype(np.float32)
        L[int(e)] = np.where(m)[0]

    picks = []
    for t in range(3):
        eps = sorted(set(ep[tk == t].tolist()))
        for k in range(PER_TASK):
            e = int(eps[int(len(eps) * (k + 0.5) / PER_TASK)])
            a = A[e]
            w = np.where(np.abs(a[:, 0] - a[0, 0]) > THRESH)[0]
            if len(w) == 0 or int(w[0]) + 49 >= len(a):
                continue
            picks.append((t, e, int(L[e][int(w[0])]), int(w[0])))

    ds = LeRobotDataset("reyqiao/lerobot_rey_toy_all", root=DATA)
    print(f"[res] {len(picks)} 帧, scales={SCALES}, URL={URL}")
    print(f"[res] 模型内部始终把内容缩到 224x168；scale<0.35 才是真的低于它")
    print(f"[res]   scale 1.000 -> 640x480 (原生)      0.350 -> 224x168 (=模型固有分辨率)")
    for s in SCALES:
        print(f"[res]   {s:5.3f} -> {int(640*s)}x{int(480*s)}")

    res = {s: [] for s in SCALES}
    for t, e, gi, t0 in picks:
        it = ds[gi]
        st = [float(x) for x in np.asarray(it["observation.state"], np.float32).reshape(-1)]
        side = img_u8(it["observation.images.side"])
        eye = img_u8(it["observation.images.eye_in_hand"])
        a = A[e]
        gidx_c49 = float(a[t0 + 49, 0])
        for s in SCALES:
            d = requests.post(URL + "/infer", json={
                "side": b64j(down(side, s)), "eye_in_hand": b64j(down(eye, s)),
                "state": st, "task": TASKS[t]}, timeout=180).json()
            res[s].append((float(np.asarray(d["action"], np.float32)[-10:].mean(0)[0]), gidx_c49, t0))
        print(f"   ... {t0} done", flush=True)

    print("\n[res] ===== 分辨率扫描 =====")
    print(f"  {'scale':>7s} {'实际尺寸':>10s}  与目标方位 r    MAE vs GT(t0+49)")
    allgt = np.array([x[1] for x in res[SCALES[0]]])
    for s in SCALES:
        pr = np.array([x[0] for x in res[s]])
        r = np.corrcoef(pr, allgt)[0, 1]
        mae = np.abs(pr - allgt).mean()
        print(f"  {s:7.3f} {int(640*s):4d}x{int(480*s):<4d}   {r:+12.3f}    {mae:12.2f} 度")

    # 惰性基线
    base = []
    for t, e, gi, t0 in picks:
        a = A[e]
        base.append((float(a[t0, 0]), float(a[t0 + 49, 0])))
    b = np.array(base)
    print(f"\n  惰性基线『停在 t0』MAE = {np.abs(b[:,0]-b[:,1]).mean():.2f} 度")
    print(f"  『永远预测 0』  MAE = {np.abs(allgt).mean():.2f} 度")


if __name__ == "__main__":
    main()
