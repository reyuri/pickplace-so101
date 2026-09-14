#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""云端 pi0.5 **LoRA** 双相机推理服务：POST /infer 返回 50 步 action chunk，端口 6030。

线协议与 serve_infer_dual.py / serve_infer_octo_redbox.py **完全一致**，所以
drive_so101_dual.py 只要在 URLS 里加一条 "PI05": "http://127.0.0.1:6030" 即可对接：
  请求  {side, eye_in_hand, task, state}      (side/eye_in_hand 是 cv2 BGR 编码的 JPEG base64)
  响应  {action: [[6 floats] x 50], infer_ms, ...}

═══ 加载配方（2026-09-14 实测踩出来的，照抄别再动）═══
我们的 ckpt 是 **LoRA/PEFT** 目录：只有 adapter_model.safetensors，**没有 model.safetensors**。

  ✗ 错法：PI05Policy.from_pretrained(ckpt)
      -> lerobot 找不到 model.safetensors，**静默返回随机初始化模型**，
         日志里只有一行 "Returning model without loading pretrained weights"，退出码 0。
         这是本项目遇到的**最恶劣的静默失败**：看起来一切正常，动作全是垃圾。

  ✓ 正法：走 lerobot 自己的 make_policy（factory.py:381-412 的 PEFT 分支）
      cfg = PreTrainedConfig.from_pretrained(ckpt)   # 我们的 2 相机 config
      cfg.pretrained_path, cfg.use_peft = ckpt, True
      policy = make_policy(cfg, ds_meta=ds.meta)     # -> PeftModel(基座 + adapter)

  另一个必踩的坑：**别用基座的 config 建模型**。PI05Policy.from_pretrained(pi05_base)
  会用基座的 3 相机（base_0_rgb/left_wrist_0_rgb/right_wrist_0_rgb）建模型，
  而我们微调的是 2 相机（side/eye_in_hand）—— 键名对不上会
      ValueError: All image features are missing from the batch
  make_policy 用 ds_meta 的 features 覆盖 cfg.input_features，正好解决。

═══ 启动时的四道硬闸门（任一不过就拒绝启动）═══
这是对上面那次「静默加载随机权重」的直接防御 —— 宁可起不来，也不要拿着随机模型上真机。
  A  基座权重真加载：与 pi05_base/model.safetensors 逐张量比对
  A2 模型相机键 = side/eye_in_hand
  B  adapter 真生效：lora_B 非零（PEFT 把 lora_B 零初始化，训练后应非零）
  C  动作量纲：跑一次预热推理，输出必须是 M100 度而非 [-1,1]

═══ 三处静默失败（计划点名，务必照做）═══
  1. predict_action_chunk 输出在**归一化空间**，必须**逐时间步**喂 postprocessor 才反归一化
     （policy_server.py:368-378 专门为此写了 for 循环）。整块喂 -> 动作幅度缩到 [-1,1]。
  2. 图像必须 float32 [0,1] HWC-RGB。传 uint8 0-255 -> 近乎全白输入且不报错。
  3. task="" 会静默退化成 "Task: , State: ...;" 照常出动作 —— 这里加硬断言拒绝空 task。

用法：
  python serve_infer_pi05.py --port=6030 \
      --ckpt=/root/autodl-tmp/outputs/pi05_lora_r16/checkpoints/018000/pretrained_model \
      --ds-root=/root/autodl-tmp/lerobot_rey_toy_all
"""
import argparse
import base64
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import torch

import lerobot.policies.pi05  # noqa: F401  注册 'pi05'
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from safetensors import safe_open

DEFAULT_TASK = "Pick up the tree and place it into the box"
G = {"lock": threading.Lock()}


def _to_tensor(hwc_u8):
    """HWC uint8 RGB -> float32 [1,3,H,W] 0..1。

    与 lerobot `prepare_observation_for_inference` 对 uint8 的处理逐字一致
    （/255 -> permute(2,0,1) -> unsqueeze(0)）。不能省 /255：_preprocess_images
    只做 cast dtype 不做归一化（modeling_pi05.py:1199-1200）再 *2-1（:1222）。
    """
    return torch.from_numpy(hwc_u8).permute(2, 0, 1).float().unsqueeze(0) / 255.0


def decode_rgb(b64):
    """请求里的图像是 cv2 BGR 编码的 JPEG -> 解出来是 BGR -> 转成 RGB（只翻这一次！）。

    历史上本项目在这里栽过：serve 端**多翻**一次通道，导致模型一直在红蓝对调下跑。
    别加任何额外的 [:, :, ::-1]。
    """
    buf = np.frombuffer(base64.b64decode(b64), np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("无法解码图片")
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def load_policy(ckpt, ds_root, repo_id, device):
    """组装 pi05 LoRA 策略并逐条把关。任何一道不过就抛异常（拒绝带病启动）。"""
    from peft import PeftConfig

    print(f"[load] ckpt = {ckpt}", flush=True)
    ds = LeRobotDataset(repo_id, root=ds_root)
    print(f"[load] dataset len = {len(ds)}", flush=True)

    peft_cfg = PeftConfig.from_pretrained(ckpt)
    base_path = peft_cfg.base_model_name_or_path
    print(f"[load] adapter 声明的基座 = {base_path}", flush=True)
    if not base_path or not os.path.isdir(base_path):
        raise SystemExit(f"基座路径无效（不是本地目录）: {base_path!r}")
    st_path = os.path.join(base_path, "model.safetensors")
    if not os.path.exists(st_path):
        raise SystemExit(f"基座目录里没有 model.safetensors: {st_path}")

    cfg = PreTrainedConfig.from_pretrained(ckpt)
    if not isinstance(cfg, PI05Config):
        raise SystemExit(f"ckpt config 不是 PI05Config 而是 {type(cfg).__name__}")
    cfg.pretrained_path = ckpt
    cfg.use_peft = True            # <- 触发 make_policy 的 PEFT 分支，少了这行就会去建全新模型
    cfg.device = device

    policy = make_policy(cfg, ds_meta=ds.meta)
    print(f"[load] make_policy -> {type(policy).__name__}", flush=True)

    # ── 闸门 A：基座权重真加载 ────────────────────────────────
    # 注意 action_in_proj 本身是 LoRA 目标，被包装后原名会变成 ...base_layer.weight
    inner = getattr(policy, "base_model", policy)
    inner = getattr(inner, "model", inner)
    got = dict(inner.named_parameters())
    ref_key = "action_in_proj.weight"
    cands = ([n for n in got if n.endswith(ref_key)]
             or [n for n in got if "action_in_proj" in n and n.endswith(".weight")])
    if not cands:
        raise SystemExit(f"[闸门A] 找不到 action_in_proj 权重，参数名样例 {list(got)[:3]}")
    with safe_open(st_path, "pt") as f:
        ref = f.get_tensor(ref_key)
    mine = got[cands[0]].detach().cpu()
    if ref.shape != mine.shape or not torch.allclose(ref.float(), mine.float(), atol=1e-5):
        raise SystemExit(f"[闸门A 失败] 基座权重没装上！{cands[0]} 与 {ref_key} 不一致"
                         f"（可能是随机初始化）")
    print(f"[gate] OK A 基座权重真加载 ({cands[0]} max|Δ|="
          f"{(ref.float()-mine.float()).abs().max().item():.2e})", flush=True)

    # ── 闸门 A2：相机键 ──────────────────────────────────────
    cam = sorted(k for k in cfg.input_features if "image" in k)
    if cam != ["observation.images.eye_in_hand", "observation.images.side"]:
        raise SystemExit(f"[闸门A2 失败] 模型相机键是 {cam}，期望"
                         f" ['observation.images.eye_in_hand', 'observation.images.side']"
                         f"（很可能是误用了基座的 3 相机 config）")
    print(f"[gate] OK A2 相机键 = {cam}", flush=True)

    # ── 闸门 B：adapter 真生效 ───────────────────────────────
    lb = [p for n, p in policy.named_parameters() if "lora_B" in n]
    if not lb:
        raise SystemExit("[闸门B 失败] 一个 lora_B 都没找到，adapter 没挂上")
    nz = sum(1 for p in lb if p.abs().sum().item() > 0)
    tot = sum(p.abs().sum().item() for p in lb)
    if nz < len(lb) * 0.9 or tot <= 0:
        raise SystemExit(f"[闸门B 失败] lora_B 几乎全零（{nz}/{len(lb)}, Σ={tot}）"
                         f"—— PEFT 零初始化态，说明 adapter 没加载")
    print(f"[gate] OK B adapter 真生效 (lora_B 非零 {nz}/{len(lb)}, Σ|B|={tot:.2f})", flush=True)

    pre, post = make_pre_post_processors(
        cfg, pretrained_path=ckpt,
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": device}},
    )
    policy = policy.to(device).eval()
    print(f"[load] 上卡完成 pre {len(pre.steps)} 步 / post {len(post.steps)} 步", flush=True)
    return policy, pre, post, cfg


def run_infer(policy, pre, post, obs):
    """一次推理。obs 是未批处理的扁平 dict，pipeline 会自己 to_transition/to_batch。"""
    # pi05 的 flow-matching 头每次采样新噪声, 默认**不固定种子**(与训练/部署惯例一致)。
    # 后果: 同一帧同一输入, 相邻两次的预测能差 6 度以上 —— 所以**别拿单次读数做结论**,
    # 横向比较要么多跑几次取统计, 要么显式 --seed 固定。
    if SEED is not None:
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    o = pre(obs)
    with torch.no_grad():
        chunk = policy.predict_action_chunk(o)          # (1, 50, 6) 归一化空间
    # ⚠️ 必须逐时间步喂 postprocessor：它期望 (B, action_dim)，
    #    整块 (B,T,dim) 过会静默得到缩在 [-1,1] 的动作
    outs = [post(chunk[:, i, :]) for i in range(chunk.shape[1])]
    return torch.stack(outs, dim=1).squeeze(0).cpu().numpy()   # (50, 6) M100 度


def warmup(policy, pre, post, device):
    """显式预热一次：flow matching + 2B PaliGemma 首调有冷启动，别让真机第一次抓取吃这个延迟。"""
    t0 = time.time()
    dummy = np.zeros((480, 640, 3), np.uint8)
    obs = {
        "observation.images.side": _to_tensor(dummy),
        "observation.images.eye_in_hand": _to_tensor(dummy),
        "observation.state": torch.zeros(1, 6),
        "task": [DEFAULT_TASK],
    }
    a = run_infer(policy, pre, post, obs)
    dt = time.time() - t0
    print(f"[warmup] 预热完成 {dt:.2f}s  输出 shape={a.shape} "
          f"range[{a.min():.2f},{a.max():.2f}]", flush=True)
    if a.shape != (50, 6):
        raise SystemExit(f"[闸门C 失败] 预热输出 shape={a.shape}，期望 (50, 6)")
    if float(np.abs(a).max()) < 1.5:
        raise SystemExit(f"[闸门C 失败] 输出落在 [-1,1]（max={np.abs(a).max():.3f}）"
                         f"—— postprocessor 没生效，动作没反归一化")
    print(f"[gate] OK C 动作量纲是 M100 度", flush=True)
    return dt


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, {"ok": True, "state": "ready", "device": DEVICE,
                         "model": "PI05", "ckpt": CKPT})

    def do_POST(self):
        if self.path != "/infer":
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n))
            task = data.get("task")
            # 计划点名的静默失败 #3：task="" 会退化成 "Task: , State: ..." 照常出动作
            if not isinstance(task, str) or not task.strip():
                raise ValueError(f"task 必须是非空字符串，收到 {task!r}")
            obs = {
                "observation.images.side": _to_tensor(decode_rgb(data["side"])),
                "observation.images.eye_in_hand": _to_tensor(decode_rgb(data["eye_in_hand"])),
                "observation.state": torch.from_numpy(
                    np.asarray(data["state"], np.float32)).view(1, -1),
                "task": [task],
            }
            t0 = time.time()
            with G["lock"]:
                action = run_infer(POLICY, PRE, POST, obs)
            dt = (time.time() - t0) * 1000
            self._send(200, {
                "action": action.tolist(),
                "action_dim": int(action.shape[1]),
                "n_steps": int(action.shape[0]),
                "infer_ms": round(dt, 1),
                "model": "PI05",
                "task": task,
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send(500, {"error": str(e)})

    def log_message(self, *a):  # quiet
        pass


CKPT = None
POLICY = PRE = POST = None
DEVICE = None
SEED = None


def main():
    global CKPT, POLICY, PRE, POST, DEVICE, SEED
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=6030)
    ap.add_argument("--ckpt",
                    default="/root/autodl-tmp/outputs/pi05_lora_r16/checkpoints/018000/pretrained_model")
    ap.add_argument("--ds-root", default="/root/autodl-tmp/lerobot_rey_toy_all")
    ap.add_argument("--repo-id", default="reyqiao/lerobot_rey_toy_all")
    ap.add_argument("--skip-gates", action="store_true",
                    help="跳过四道硬闸门（仅调试用，真机千万别加）")
    ap.add_argument("--seed", type=int, default=None,
                    help="固定 flow-matching 采样种子（默认不固定=真机自然行为；调试复现时才加）")
    a = ap.parse_args()
    if a.skip_gates:
        raise SystemExit("拒绝以 --skip-gates 启动：本项目的静默加载事故就是这么来的")
    SEED = a.seed

    CKPT = a.ckpt
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    POLICY, PRE, POST, cfg = load_policy(a.ckpt, a.ds_root, a.repo_id, DEVICE)
    print(f"[load] 加载耗时 {time.time()-t0:.1f}s", flush=True)
    warmup(POLICY, PRE, POST, DEVICE)

    srv = ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    print(f"[serve] listening http://0.0.0.0:{a.port}  /infer  model=PI05  ckpt={a.ckpt}",
          flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
