import base64, io, json, glob, urllib.request
import numpy as np
from PIL import Image

URL = "http://127.0.0.1:6020/infer"
TASKS = {"tree": "Pick up the tree and place it into the box",
         "ball": "Pick up the ball and place it into the box",
         "elephant": "Pick up the elephant and place it into the box"}

r = json.loads(urllib.request.urlopen(URL.replace("/infer", "/"), timeout=30).read())
print("[GET /] ckpt=%s step-loaded -- %s" % (r.get("ckpt"), r.get("state")))

d = np.load(sorted(glob.glob("/root/autodl-tmp/octo_frames/tree/ep_*.npz"))[0])
f = min(10, d["side"].shape[0] - 1)
side = np.ascontiguousarray(d["side"][f][:, :, :3].astype(np.uint8))
wrist = np.ascontiguousarray(d["wrist"][f][:, :, :3].astype(np.uint8))

def b64(img):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()

s, w = b64(side), b64(wrist)

def call(task, reset=True):
    body = json.dumps({"side": s, "eye_in_hand": w, "task": task, "reset": reset}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=180).read())

out = {}
for k, t in TASKS.items():
    r = call(t)
    a = np.array(r["action"])
    out[k] = a
    print("  %-9s infer_ms=%-7s 首步(6关节)= %s" % (k, r["infer_ms"], np.round(a[0], 2)))

print()
print("  两两首步最大差: tree-vs-ball=%.2f  tree-vs-elephant=%.2f  ball-vs-elephant=%.2f"
      % (np.abs(out["tree"][0] - out["ball"][0]).max(),
         np.abs(out["tree"][0] - out["elephant"][0]).max(),
         np.abs(out["ball"][0] - out["elephant"][0]).max()))
print("  整块(4x6)最大差: %.2f   动作幅度 max|a|: %.2f"
      % (max(np.abs(out[a] - out[b]).max() for a in out for b in out),
         max(np.abs(v).max() for v in out.values())))
