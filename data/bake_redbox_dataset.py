# -*- coding: utf-8 -*-
"""A2 红框数据集烘焙: 把目标框(conf>=0.3)烙进 side 视频,生成 lerobot_rey_toy_all_redbox。

- 仅重编码 side 视频(h264/30fps/640x480, 帧数不变); eye_in_hand / meta / data / tasks 原样拷贝。
- 每帧: sidecar 有目标框且 conf>=0.3 -> 画 1 个粗红框 + 小类名; 否则原图不动。
- 唯一规则: 只画目标框, 无任何其他检测线(干扰全去)。

跑法(服务器): /root/autodl-tmp/venv_yolo/bin/python /root/autodl-tmp/bake_redbox_dataset.py
产物: /root/autodl-tmp/lerobot_rey_toy_all_redbox/
"""
import glob
import json
import os
import shutil

import av
import cv2

SRC = "/root/autodl-tmp/lerobot_rey_toy_all"
DST = "/root/autodl-tmp/lerobot_rey_toy_all_redbox"
SIDEBOX = "/root/autodl-tmp/yolo_target_boxes"
THR = 0.3
FPS = 30
RED = (0, 0, 255)
YOLO2NAME = {0: "elephant", 1: "tree", 2: "ball"}

# ---- 1. 建 DST + 拷贝非 side 内容 ----
os.makedirs(DST, exist_ok=True)
for sub in ("meta", "data"):
    d = os.path.join(DST, sub)
    if not os.path.exists(d):
        shutil.copytree(os.path.join(SRC, sub), d)
os.makedirs(os.path.join(DST, "videos"), exist_ok=True)
dseye = os.path.join(DST, "videos", "observation.images.eye_in_hand")
if not os.path.exists(dseye):
    shutil.copytree(os.path.join(SRC, "videos", "observation.images.eye_in_hand"), dseye)
oside = os.path.join(DST, "videos", "observation.images.side", "chunk-000")
os.makedirs(oside, exist_ok=True)
print("[bake] DST 骨架就绪:", DST, flush=True)

ep_files = sorted(glob.glob(SIDEBOX + "/ep_*.json"))


def encode_stream(frames_iter, out_path):
    """frames_iter 逐个产出 BGR ndarray;流式写 h264,内存恒定,帧数不丢失。"""
    out = av.open(out_path, "w")
    st = out.add_stream("h264", rate=FPS)
    st.width = 640
    st.height = 480
    st.pix_fmt = "yuv420p"
    st.options = {"preset": "veryfast", "crf": "20"}
    for fr in frames_iter:
        vf = av.VideoFrame.from_ndarray(fr, format="bgr24")
        for pkt in st.encode(vf):
            out.mux(pkt)
    for pkt in st.encode(None):
        out.mux(pkt)
    out.close()


drawn = 0
no_box = 0
for idx, jf in enumerate(ep_files):
    ep = int(os.path.basename(jf)[3:6])
    box = json.load(open(jf))
    cap = cv2.VideoCapture(f"{SRC}/videos/observation.images.side/chunk-000/file-{ep:03d}.mp4")
    cnt = {"drawn": 0, "nobox": 0}

    def gen():
        fi = 0
        while True:
            ret, fr = cap.read()
            if not ret or fr is None:
                break
            b = box.get(str(fi))
            if b is not None and b[5] >= THR:
                _, cx, cy, bw, bh, conf = b
                h, w = fr.shape[:2]
                x1 = int((cx - bw / 2) * w); y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w); y2 = int((cy + bh / 2) * h)
                cv2.rectangle(fr, (x1, y1), (x2, y2), RED, 3)
                cv2.putText(fr, f"{YOLO2NAME[int(b[0])]} {conf:.2f}", (x1, max(y1 - 6, 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)
                cnt["drawn"] += 1
            else:
                cnt["nobox"] += 1
            yield fr
            fi += 1

    encode_stream(gen(), os.path.join(oside, f"file-{ep:03d}.mp4"))
    cap.release()
    drawn += cnt["drawn"]
    no_box += cnt["nobox"]
    if (idx + 1) % 15 == 0 or idx == len(ep_files) - 1:
        print(f"[bake] {idx+1}/{len(ep_files)} ep{ep} 画框累计{drawn} 无框累计{no_box}", flush=True)

print(f"\n[bake] DONE: {len(ep_files)} ep, 烙框 {drawn} 帧, 无框(保原图) {no_box} 帧", flush=True)
print("[bake] 数据集:", DST)
