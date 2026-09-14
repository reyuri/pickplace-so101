#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""核心问题的正确问法：**在手臂开始朝目标转的那一刻**，模型有没有转对方向？

前两次都栽在取帧上：
  - 取 episode 40% 处 -> 已快到位，目标早定死，对语言不利
  - 取 起始帧 8 -> 落在**静止死区**（ep0 前 75 帧完全不动），GT 就是"别动"，比什么都错
正确的位置是 **t0 = 手臂 j1 首次离开初始值 >5 度的那一帧**：
此时目标已定、GT 正在转，而 chunk 的 50 步正好覆盖"转到抓取"。

必须先做前检：若 GT 自己在 t0+49 的 j1 与抓取时刻 j1 都不相关，说明 t0 没选对，结论作废。
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
THRESH = float(os.environ.get("PI05_THRESH", "5"))
NAMES = ["j1", "j2", "j3", "j4", "j5", "grip"]
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


def main():
    files = sorted(glob.glob(f"{DATA}/data/**/*.parquet", recursive=True))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    ep = df["episode_index"].to_numpy()
    tk = df["task_index"].to_numpy()
    A, S, L = {}, {}, {}
    for e in sorted(set(ep.tolist())):
        m = ep == e
        A[int(e)] = np.stack(df.loc[m, "action"].to_numpy()).astype(np.float32)
        S[int(e)] = np.stack(df.loc[m, "observation.state"].to_numpy()).astype(np.float32)
        L[int(e)] = np.where(m)[0]

    picks = []
    for t in range(3):
        eps = sorted(set(ep[tk == t].tolist()))
        for k in range(PER_TASK):
            e = int(eps[int(len(eps) * (k + 0.5) / PER_TASK)])
            a = A[e]
            dev = np.abs(a[:, 0] - a[0, 0])
            w = np.where(dev > THRESH)[0]
            if len(w) == 0:
                continue
            t0 = int(w[0])
            if t0 + 49 >= len(a):
                continue
            picks.append((t, e, int(L[e][t0]), t0))

    ds = LeRobotDataset("reyqiao/lerobot_rey_toy_all", root=DATA)
    print(f"[cm] {len(picks)} 帧 @ t0(手臂 j1 首次偏离初始 >{THRESH}度), URL={URL}")

    rows = []
    for t, e, gi, t0 in picks:
        it = ds[gi]
        st = np.asarray(it["observation.state"], np.float32).reshape(-1)
        a, s = A[e], S[e]
        # 抓取时刻
        op = np.where(a[:, 5] > 30)[0]
        gidx = int(op[0] + np.where(a[int(op[0]):, 5] < 5)[0][0]) if len(op) else None
        out = {}
        for lab, task in (("true", TASKS[t]), ("wrong", TASKS[(t + 1) % 3])):
            d = requests.post(URL + "/infer", json={
                "side": b64j(img_u8(it["observation.images.side"])),
                "eye_in_hand": b64j(img_u8(it["observation.images.eye_in_hand"])),
                "state": [float(x) for x in st], "task": task}, timeout=180).json()
            out[lab] = np.asarray(d["action"], np.float32)
        ct, cw = out["true"], out["wrong"]
        rows.append({
            "task": t, "ep": e, "t0": t0,
            "gt_now": float(a[t0, 0]),
            "gt_c49": float(a[min(t0 + 49, len(a) - 1), 0]),
            "gt_grasp": float(a[gidx, 0]) if gidx is not None else np.nan,
            "pr_end": float(ct[-10:].mean(0)[0]), "pr_last": float(ct[-1, 0]),
            "pw_end": float(cw[-10:].mean(0)[0]),
            "gt_c49_j2": float(a[min(t0 + 49, len(a) - 1), 1]),
            "pr_end_j2": float(ct[-10:].mean(0)[1]),
            "st_j2": float(st[1]),
        })

    R = pd.DataFrame(rows)
    ok = ~np.isnan(R.gt_grasp.to_numpy())

    print("\n[cm] 前检：GT 的 j1 在 t0+49 与『抓取时刻 j1』的相关")
    r = np.corrcoef(R.gt_c49[ok], R.gt_grasp[ok])[0, 1]
    print(f"   r = {r:+.3f}   (目标方位 std {R.gt_grasp[ok].std():5.2f}, "
          f"t0 时 j1 std {R.gt_now[ok].std():5.2f})")
    print("   -> 必须明显为正，否则 t0 没选对，下面的结论作废")

    print("\n[cm] 逐条明细")
    print("  任务        ep   t0   当时j1   目标方位@grasp   模型末10步j1  (换错指令)")
    for _, x in R.iterrows():
        print(f"  {TASKS[int(x.task)]:>8s} {int(x.ep):5d} {int(x.t0):4d} "
              f"{x.gt_now:8.2f} {x.gt_grasp:14.2f} {x.pr_end:14.2f} {x.pw_end:11.2f}")

    print("\n[cm] ===== 模型有没有转对方向 =====")
    for lab, col in (("真指令", "pr_end"), ("错指令", "pw_end")):
        r1 = np.corrcoef(R.gt_grasp[ok], R[col][ok])[0, 1]
        r2 = np.corrcoef(R.gt_c49[ok], R[col][ok])[0, 1]
        print(f"  {lab}: 与『目标方位@grasp』r = {r1:+.3f}   与『t0+49 的 GT』r = {r2:+.3f}"
              f"   MAE vs t0+49 = {np.abs(R[col][ok] - R.gt_c49[ok]).mean():6.2f} 度")
    lazy = np.abs(R.gt_now[ok] - R.gt_c49[ok]).mean()
    print(f"  惰性基线『j1 停在 t0 不动』 MAE vs t0+49 = {lazy:6.2f} 度")
    print(f"  语言影响: |真-错| 平均 = {(R.pr_end - R.pw_end).abs().mean():.2f} 度")

    print("\n[cm] j2（伸出/高低）同样看一眼")
    rr = np.corrcoef(R.gt_c49_j2[ok], R.pr_end_j2[ok])[0, 1]
    lz = np.abs(R.st_j2[ok] - R.gt_c49_j2[ok]).mean()
    print(f"  模型 vs GT(t0+49): r = {rr:+.3f}, MAE = "
          f"{np.abs(R.pr_end_j2[ok] - R.gt_c49_j2[ok]).mean():6.2f} 度; 惰性基线 {lz:6.2f} 度")


if __name__ == "__main__":
    main()
