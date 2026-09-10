#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dual_camera_viewer.py
--------------------
实时展示摄像头画面（支持 1 路或 2 路）。

用法（用 lerobot 环境的 python 运行）：
    python dual_camera_viewer.py               # 自动探测，显示前 1~2 个可用的摄像头
    python dual_camera_viewer.py 1             # 只显示 1 路（idx=1）
    python dual_camera_viewer.py 1 0           # 2 路并排，与真机脚本默认槽位一致（1=侧视, 0=手眼）
    python dual_camera_viewer.py --list        # 只列出能读到画面的摄像头索引

⚠️ 索引会漂移，别背死数字，每次目视认画面：
    2026-09-10 实测:  idx0=`USB Camera`=手眼(俯视夹爪) / idx1=`Brio 90`=侧视(桌面全景) / idx2=笔记本内置
    历史(2026-09-04): idx0=笔记本 / idx1=侧视 / idx2=手眼   ← 此映射已失效，别再用
驱动 drive_so101_dual.py 默认槽位 = --cam-side 1 --cam-eye 0，漂了就用这两个参数覆盖。

键位：q 或 ESC 退出

说明：
    - 给 1 个索引 -> 只显示该相机；给 2 个索引 -> 水平并排显示两个相机。
    - 两个相机的曝光属性 OpenCV 无法可靠控制，故原样显示，不做后处理。
    - 为抗双开黑屏，强制 MJPG、保持自动曝光、加预热。
"""

import sys
import time
import cv2
import numpy as np

BACKEND = cv2.CAP_DSHOW
MAX_PROBE = 6
W, H = 640, 480


def probe(max_idx: int = MAX_PROBE) -> list:
    """返回能打开并读到画面的摄像头索引。"""
    cams = []
    for i in range(max_idx + 1):
        cap = cv2.VideoCapture(i, BACKEND)
        ok, frame = cap.read()
        if ok and frame is not None:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cams.append(i)
            print(f"  [found] cam idx={i}  size={w}x{h}")
        cap.release()
    return cams


def open_cam(idx: int):
    """打开单个摄像头，抗黑屏初始化。失败返回 None。"""
    cap = cv2.VideoCapture(idx, BACKEND)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # MJPG 编码，双开更稳；开启自动曝光；预热等曝光收敛(2~5s)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)   # 必须 1.0 才真正开自动曝光; 0.25 会造成全白
    time.sleep(2.5)
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        print(f"  [!] cam idx={idx} open FAILED")
        return None
    for _ in range(3):
        cap.read()
    print(f"  [ok] cam idx={idx}  usable")
    return cap


def main():
    args = [a for a in sys.argv[1:]]

    if "--list" in args:
        print(f"扫描可用摄像头索引 0..{MAX_PROBE}:")
        found = probe()
        print("可用摄像头索引:", found if found else "无")
        return

    indexes = []
    for a in args:
        if a.isdigit():
            indexes.append(int(a))

    if not indexes:
        print("自动探测可用摄像头...")
        found = probe()
        if not found:
            print("没有可用的摄像头。")
            return
        indexes = found[:2]

    # 去重、保序
    seen = set()
    uniq = []
    for i in indexes:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    indexes = uniq[:2]  # 只支持 1 或 2 路

    if len(indexes) < 1:
        print("未指定摄像头。")
        return

    if len(indexes) == 2:
        print(f"开始展示 2 路摄像头: idx={indexes[0]} 与  idx={indexes[1]}  （按 q 退出）")
    else:
        print(f"开始展示 1 路摄像头: idx={indexes[0]}  （按 q 退出）")

    caps = []
    for i in indexes:
        c = open_cam(i)
        caps.append((i, c))

    caps = [(i, c) for i, c in caps if c is not None]
    if not caps:
        print("所有摄像头都无法打开，可能是被其他程序占用。")
        return
    if len(caps) == 1:
        print("[!] 只成功打开 1 路。")

    window_name = " | ".join(f"Cam {i}" for i, _ in caps)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            frames = []
            for i, c in caps:
                ok, f = c.read()
                if ok and f is not None:
                    label = f"idx={i}"
                    cv2.putText(f, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 0), 2)
                    frames.append(f)

            if not frames:
                print("[!] 无法读取画面，摄像头可能被占用。")
                break

            if len(frames) == 1:
                combo = frames[0]
            else:
                target_h = min(f.shape[0] for f in frames)
                resized = []
                for f in frames:
                    scale = target_h / f.shape[0]
                    resized.append(cv2.resize(f, (int(f.shape[1] * scale), target_h)))
                combo = np.hstack(resized)

            cv2.imshow(window_name, combo)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        for _, c in caps:
            c.release()
        cv2.destroyAllWindows()
        print("已退出。")


if __name__ == "__main__":
    main()
