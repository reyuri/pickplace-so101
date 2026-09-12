# -*- coding: utf-8 -*-
"""val_corrected.py —— 改正后的选点口径: 跨 step 扫描, 三段式(批量前向版)。

为什么不能再用 VAL_loss 选点
------------------------------
`VAL_loss` 是归一化空间里**扩散噪声预测的 MSE**(乘 action_dim=7), 被高噪声档位主导,
单位不是角度, 读不出"偏几度"。实测三个 run 的 VAL 曲线全是平噪声、且与真机反相关:
  红框版 8000 VAL 最好(0.3354) -> 夹爪该闭却张; 16000 VAL 最差(0.6650) -> 真机正常。
所以选点必须换口径。

三段式口径(互相独立, 都不依赖 VAL)
----------------------------------
  ① MAE  : M100 动作 MAE, 与两个惰性基线(什么都不动 / 沿用上一帧)比。部署同款采样
           (2 帧窗口 / 纯视觉 / argmax=False / seed 42)。
  ② GRIP : 夹爪开合**时序**——首次闭合的归一化时间位置误差 / grip 轨迹相关 / 最小 grip。
           直接对应 8000 那个"该闭合时张开"的失败模式。
  ③ COND : 条件响应幅度——同 seed 配对的 Δ(相对) + 相干比 + 换 seed 噪声地板。
           三版统一走**语言**通道(同帧换 toy 名), 红框版另加**红框**通道。
           相干比是 0~1 归一化后的 `相干n`(0=纯噪声 / 1=完全相干),
           原始相干比对比纯噪声的期望是 **1.75 不是 0.25**(推导见 `_paired`)。

选点规则: ① 必须低于「沿用上一帧」; 主排序看 ②; 并列打破看 ③。

⚠️ 默认 batch=1 —— 不要为了"快"把它调大
---------------------------------------
本脚本曾改成批量前向提速，**结论是不该改**(理由见 `BATCH` 定义处)。实测:
batch=1 时 ① 与 `eval_m100.py` **逐位相同**(redbox@8000 = 12.89 / 夹爪 21.55)，
batch=16 时变成 20.47 / 30.64。批量会静默换掉一把尺子。

真正的性能瓶颈不在算力，在 IO：`octo_frames_*.npz` 是 savez_compressed 存的，
而 `np.load` 返回的 NpzFile 惰性——`d["side"]` 每访问一次重新解压 35MB(实测 0.33s,
不缓存)。旧写法按帧访问，~8 分钟/step、GPU 全程 0%。改成 `_eager` 一次性载入后，
batch=1 也只要 **~75s/step**(提速 6.4× 全来自这里)。

所以 ① 与 `eval_m100.py` 可以直接对表，历史上记过的数不用重跑。

用法
----
  P=/root/autodl-tmp/conda_envs/octo/bin/python
  $P val_corrected.py --run redbox                       # 自动扫 ckpt 目录下所有 step
  $P val_corrected.py --run base --steps 1000 8000 16000 # 指定 step
  $P val_corrected.py --run all --out /root/autodl-tmp/val_corrected.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_CACHE", "/root/autodl-tmp/hf_hub")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from octo.model.octo_model import OctoModel  # noqa: E402

# --------------------------------------------------------------------------- 配置
PLAIN_TPL = "Pick up the %s and place it into the box"
RED_TPL = "Pick up the %s in red box and place it into the box"
TOYS = ("ball", "elephant", "tree")        # 与 builder 的 sorted() 一致
VAL_LOCAL = (0, 9, 18, 27, 36)             # ep % 9 == 0，每 toy 5 条
CLOSE_TH = 10.0                            # 夹爪"闭合"阈值，同驱动 --done-grip-close
N_MAE_FRAMES = 8                           # ① 每 episode 抽几帧
N_GRIP_PTS = 24                            # ② 每 episode 抽几个时间点
N_COND_FRAMES = 2                          # ③ 每 episode 取几帧做条件对比
SEEDS = (0, 1, 2, 3)                       # ③ 配对 seed
BASE_SEED = 42                             # ① ② 部署同款 seed

# ⚠️ 批量前向会改变 ① 的数值，默认必须是 1。
# 实测(redbox@8000, 同一帧集/同一 seed)：batch=1 -> 12.89，batch=16 -> 20.47，
# 夹爪 21.55 -> 30.64。原因：batch=1 时每帧各自 `normal(PRNGKey(42),(1,4,7))`，
# **所有帧共用同一次噪声抽样**(= eval_m100.py = serve 的部署行为)；batch=B 时一次抽
# `(B,4,7)`，**各帧拿到互相独立的噪声**，误差分布随之改变。
# 部署端 serve_octo_*.py 就是「seed 42 抽一次」，所以 batch=1 才是与真机同口径的
# 那把尺子。去掉惰性解压(见 `_eager`)后 batch=1 已只要 ~75s/step，批量不再有必要。
BATCH = 1                                  # 批量前向上限(默认 1，别乱改，见上)

RUNS = {
    "base": dict(ckpt="/root/autodl-tmp/octo_lora_fix",
                 frames="/root/autodl-tmp/octo_frames", tpl=PLAIN_TPL, box=False),
    "infonce": dict(ckpt="/root/autodl-tmp/octo_lora_infonce_well",
                    frames="/root/autodl-tmp/octo_frames", tpl=PLAIN_TPL, box=False),
    "redbox": dict(ckpt="/root/autodl-tmp/octo_lora_redbox",
                   frames="/root/autodl-tmp/octo_frames_redbox", tpl=RED_TPL, box=True),
}
BOX_JSON = "/root/autodl-tmp/_probe_boxes.json"
RED = np.array([255, 0, 0], dtype=np.uint8)
BOX_TH = 2


# --------------------------------------------------------------------------- 基础
def to128(x):
    return np.asarray(Image.fromarray(np.array(x)).resize((128, 128)))


def denorm_stats(model):
    """6 维 M100 统计量, 尾部补齐第 7 维(gripper 占 dim5, dim6 是 pad_action7 的哑列)。"""
    st = model.dataset_statistics["action"]
    amean = np.concatenate([np.asarray(st["mean"], np.float32), [0.0]])
    astd = np.concatenate([np.asarray(st["std"], np.float32), [1.0]])
    return amean, astd


def _encode(model, strs):
    return {"language_instruction": model.text_processor.encode(list(strs))}


def sample_win(model, ds, fs, strs, seed):
    """2 帧窗口 [f-1,f] 的批量采样。返回 (N,4,7) 归一化动作。"""
    out = []
    for i in range(0, len(ds), BATCH):
        sl = slice(i, i + BATCH)
        pri = np.stack([d["side"][f - 1:f + 1] for d, f in zip(ds[sl], fs[sl])])
        wri = np.stack([np.stack([to128(x) for x in d["wrist"][f - 1:f + 1]])
                        for d, f in zip(ds[sl], fs[sl])])
        obs = {"image_primary": jnp.asarray(pri, jnp.uint8),
               "image_wrist": jnp.asarray(wri, jnp.uint8),
               "timestep_pad_mask": jnp.ones((len(ds[sl]), 2), dtype=bool)}
        a = model.sample_actions(obs, _encode(model, strs[sl]), argmax=False,
                                 rng=jax.random.PRNGKey(seed))
        out.append(np.asarray(a).astype(np.float64))
    return np.concatenate(out, 0)


def sample_frame(model, sides, wrists, strs, seed):
    """单帧复制两遍(box 探针口径, 同 probe_cross.py)。返回 (N,4,7)。"""
    out = []
    for i in range(0, len(sides), BATCH):
        sl = slice(i, i + BATCH)
        pri = np.stack([np.stack([s, s]) for s in sides[sl]])
        wri = np.stack([np.stack([w, w]) for w in wrists[sl]])
        obs = {"image_primary": jnp.asarray(pri, jnp.uint8),
               "image_wrist": jnp.asarray(wri, jnp.uint8),
               "timestep_pad_mask": jnp.ones((len(sides[sl]), 2), dtype=bool)}
        a = model.sample_actions(obs, _encode(model, strs[sl]), argmax=False,
                                 rng=jax.random.PRNGKey(seed))
        out.append(np.asarray(a).astype(np.float64))
    return np.concatenate(out, 0)


def stamp(img, cxcywh):
    """2px 纯红框, 与 serve_octo_redbox.py 的 stamp 逐像素等价。原地改。"""
    H, W = img.shape[:2]
    cx, cy, bw, bh = cxcywh
    x1 = max(0, min(W - 1, int(round((cx - bw / 2) * W))))
    x2 = max(0, min(W - 1, int(round((cx + bw / 2) * W))))
    y1 = max(0, min(H - 1, int(round((cy - bh / 2) * H))))
    y2 = max(0, min(H - 1, int(round((cy + bh / 2) * H))))
    if x2 <= x1 or y2 <= y1:
        return img
    img[y1:y1 + BOX_TH, x1:x2 + 1] = RED
    img[max(0, y2 - BOX_TH + 1):y2 + 1, x1:x2 + 1] = RED
    img[y1:y2 + 1, x1:x1 + BOX_TH] = RED
    img[y1:y2 + 1, max(0, x2 - BOX_TH + 1):x2 + 1] = RED
    return img


def _eager(path):
    """把一条 episode 全量读进内存。

    ⚠️ 必须这么做。`octo_frames_*.npz` 是 savez_compressed 存的(compress_type=8)，
    而 `np.load` 返回的 NpzFile 是**惰性**的：`d["side"]` 每访问一次就重新解压一整个
    34.8MB 成员, 且**不缓存**——实测 0.33s/次(side)、0.31s/次(wrist)。
    按逐项访问的写法，光解压就 ~8 分钟/step，把 GPU 完全饿死(实测 GPU util 0%、
    CPU 单核 100%)。这不是模型慢，是 IO 慢。
    """
    z = np.load(path)
    try:
        return {k: np.ascontiguousarray(z[k]) for k in ("side", "wrist", "action", "state")}
    finally:
        z.close()


def load_eps(frames_root):
    eps = []
    for toy in TOYS:
        for e in VAL_LOCAL:
            p = "%s/%s/ep_%d.npz" % (frames_root, toy, e)
            if os.path.exists(p):
                eps.append((toy, e, _eager(p)))
            else:
                print("  [skip] %s" % p, flush=True)
    n = sum(len(d["side"]) for _, _, d in eps)
    print("  已全量载入 %d 条 episode / %d 帧(常驻内存)" % (len(eps), n), flush=True)
    return eps


# --------------------------------------------------------------------------- ① MAE
def part_mae(model, amean, astd, eps, tpl, nframes):
    ds, fs, strs = [], [], []
    for toy, e, d in eps:
        n = len(d["side"])
        if n < 7:
            continue
        for f in np.linspace(1, n - 5, nframes).astype(int):
            ds.append(d); fs.append(f); strs.append(tpl % toy)
    A = (sample_win(model, ds, fs, strs, BASE_SEED) * astd + amean)[:, :, :6]  # (N,4,6)
    errs, do, prev = [], [], []
    for i, (d, f) in enumerate(zip(ds, fs)):
        gt = d["action"][f:f + 4, :6]
        k = len(gt)
        errs.extend(np.abs(A[i][:k] - gt))
        do.extend(np.abs(d["state"][f][:6] - gt))
        prev.extend(np.abs(d["action"][f - 1][:6] - gt))
    return np.asarray(errs), np.asarray(do), np.asarray(prev)


# --------------------------------------------------------------------------- ② GRIP
def _first_close(t):
    """首次闭合的**归一化时间位置**(0~1), 长度无关。没闭合返回 nan。"""
    i = np.where(t <= CLOSE_TH)[0]
    return float(i[0]) / (len(t) - 1) if len(i) else float("nan")


def part_grip(model, amean, astd, eps, tpl, npts):
    ds, fs, strs, owner = [], [], [], []
    for j, (toy, e, d) in enumerate(eps):
        n = len(d["side"])
        if n < 7:
            continue
        for f in np.linspace(1, n - 1, min(npts, n - 1)).astype(int):
            ds.append(d); fs.append(f); strs.append(tpl % toy); owner.append(j)
    A = sample_win(model, ds, fs, strs, BASE_SEED) * astd + amean          # (N,4,7)
    rows = []
    for j, (toy, e, d) in enumerate(eps):
        idx = [i for i, o in enumerate(owner) if o == j]
        if not idx:
            continue
        pg = A[idx, 0, 5]                                   # 递推式重预测: 取每轮 chunk 第一步
        gt = np.asarray([float(d["action"][fs[i], 5]) for i in idx])
        corr = float(np.corrcoef(pg, gt)[0, 1]) if pg.std() > 1e-6 and gt.std() > 1e-6 \
            else float("nan")
        rows.append(dict(ep="%s/ep_%d" % (toy, e), t_pred=_first_close(pg), t_gt=_first_close(gt),
                         corr=corr, min_pred=float(pg.min()), min_gt=float(gt.min())))
    return rows


def grip_summary(rows):
    t_err, corr, dmin = [], [], []
    for r in rows:
        if not (np.isnan(r["t_pred"]) or np.isnan(r["t_gt"])):
            t_err.append(abs(r["t_pred"] - r["t_gt"]))
        if not np.isnan(r["corr"]):
            corr.append(r["corr"])
        dmin.append(r["min_pred"])
    return dict(timing_err=float(np.mean(t_err)) if t_err else float("nan"), n_timing=len(t_err),
                corr=float(np.mean(corr)) if corr else float("nan"),
                min_grip=float(np.mean(dmin)) if dmin else float("nan"),
                n_fail_open=int(np.sum(np.asarray(dmin) > CLOSE_TH)),   # 该闭却始终张开
                n_ep=len(rows))


# --------------------------------------------------------------------------- ③ COND
def _paired(ref, alt):
    """同 seed 配对: d_s = alt(s) - ref(s)。返回 Δ(相对) / 相干比 / 归一化相干 / 地板。

    ⚠️ 相干比的量纲坑(代码抄自 probe_cross.py:121, 但那边注释里的阈值是错的)：
        分子 = Σ_元素 mean_s(d)²            (逐元素求和)
        分母 = mean( Σ_h d² ) = horizon·σ²   (对元素取平均)
    两边的"元素数"口径不一致，于是纯噪声的期望不是 0.25 而是
        chance = N_elem/(n_seed·horizon) = (horizon·action_dim)/(n_seed·horizon)
               = action_dim/n_seed = 7/4 = 1.75    (实测 2000 次: mean 1.751)
    完全相干 = action_dim = 7.0。
    所以相干比落在 [1.75, 7.0] 区间, 不是 [0.25, 1]。这里额外给出归一化到 0~1 的
    `coh_n`(0=纯噪声, 1=完全相干), 同时保留原始 `coh` 以便和旧 probe_cross 结果对表。
    """
    n = len(SEEDS)
    D = np.stack([alt[s] - ref[s] for s in SEEDS])
    m = D.mean(0)
    a0 = ref[SEEDS[0]]
    amp = float(np.linalg.norm(m) / (np.linalg.norm(a0) + 1e-12))
    coh = float((m * m).sum()) / max(float(np.mean((D * D).sum(1))), 1e-12)
    chance, top = float(D.shape[-1]) / n, float(D.shape[-1])
    coh_n = (coh - chance) / (top - chance)
    floor = float(np.mean([np.linalg.norm(ref[b] - ref[a]) / (np.linalg.norm(ref[a]) + 1e-12)
                           for i, a in enumerate(SEEDS) for b in SEEDS[i + 1:]]))
    return amp, coh, coh_n, floor


def _cond_stats(amp, coh, coh_n, floor):
    f = lambda v: float(np.mean(v)) if v else float("nan")  # noqa: E731
    return dict(amp=f(amp), coh=f(coh), coh_n=f(coh_n), floor=f(floor), n=len(amp))


def part_cond(model, eps, tpl, box_json):
    """语言通道(三版统一) + 红框通道(仅红框版, 有真框数据时)。"""
    boxes = json.load(open(box_json)) if (box_json and os.path.exists(box_json)) else {}
    items = []
    for toy, e, d in eps:
        n = len(d["side"])
        if n < 7:
            continue
        for f in np.linspace(max(1, n // 4), n - 2, N_COND_FRAMES).astype(int):
            items.append((toy, e, d, int(f)))
    if not items:
        return {"lang": _cond_stats([], [], [], [])}
    ds = [it[2] for it in items]; fs = [it[3] for it in items]
    alt_of = {t: ("ball" if t != "ball" else "tree") for t in TOYS}
    ref = {s: sample_win(model, ds, fs, [tpl % it[0] for it in items], 1000 + s) for s in SEEDS}
    alt = {s: sample_win(model, ds, fs, [tpl % alt_of[it[0]] for it in items], 1000 + s)
           for s in SEEDS}
    la, lc, ln, lf = [], [], [], []
    for i in range(len(items)):
        a, c, cn, f_ = _paired({s: ref[s][i] for s in SEEDS}, {s: alt[s][i] for s in SEEDS})
        la.append(a); lc.append(c); ln.append(cn); lf.append(f_)
    out = {"lang": _cond_stats(la, lc, ln, lf)}

    # --- 红框通道: 同帧同指令, 只挪框 ---
    bi, bs_t, bs_b, wrs, strs = [], [], [], [], []
    for k, (toy, e, d, f) in enumerate(items):
        key = "%s/ep_%d.npz" % (toy, e)
        bx = boxes.get(key, {}).get("boxes") if boxes else None
        alt_toy = alt_of[toy]
        if not (bx and toy in bx and alt_toy in bx):
            continue
        bi.append(k)
        bs_t.append(stamp(np.ascontiguousarray(d["side"][f].copy()), bx[toy]))
        bs_b.append(stamp(np.ascontiguousarray(d["side"][f].copy()), bx[alt_toy]))
        wrs.append(to128(d["wrist"][f])); strs.append(tpl % toy)
    if bi:
        rt = {s: sample_frame(model, bs_t, wrs, strs, 2000 + s) for s in SEEDS}
        rb = {s: sample_frame(model, bs_b, wrs, strs, 2000 + s) for s in SEEDS}
        ba, bc, bn, bf = [], [], [], []
        for i in range(len(bi)):
            a, c, cn, f_ = _paired({s: rt[s][i] for s in SEEDS}, {s: rb[s][i] for s in SEEDS})
            ba.append(a); bc.append(c); bn.append(cn); bf.append(f_)
        out["box"] = _cond_stats(ba, bc, bn, bf)
    return out


# --------------------------------------------------------------------------- 单 step
def eval_step(model, cfg, eps, nframes, npts):
    amean, astd = denorm_stats(model)
    e, do, pv = part_mae(model, amean, astd, eps, cfg["tpl"], nframes)
    grip = grip_summary(part_grip(model, amean, astd, eps, cfg["tpl"], npts))
    cond = part_cond(model, eps, cfg["tpl"], BOX_JSON if cfg["box"] else None)
    return dict(mae=float(e.mean()), mae_grip=float(e[:, 5].mean()),
                mae_do=float(do.mean()), mae_prev=float(pv.mean()),
                mae_do_grip=float(do[:, 5].mean()), mae_prev_grip=float(pv[:, 5].mean()),
                grip=grip, cond=cond)


def list_steps(ckpt_dir):
    return sorted(int(n) for n in os.listdir(ckpt_dir)
                  if re.fullmatch(r"\d+", n) and os.path.isdir(os.path.join(ckpt_dir, n)))


def fmt(name, st, r):
    """相干列输出**归一化**值(0=纯噪声 / 1=完全相干)；原始相干比见 JSON 的 cond.*.coh。"""
    g, c = r["grip"], r["cond"]
    s = ("  %-8s step %-6d MAE %6.2f (不动 %5.2f / 上帧 %5.2f) | 夹爪 %6.2f | "
         "闭合时机误 %5.3f corr %6.3f 最小 %7.2f 未闭 %d/%d | 语言Δ %6.4f 相干n %5.3f 地板 %6.4f"
         % (name, st, r["mae"], r["mae_do"], r["mae_prev"], r["mae_grip"], g["timing_err"],
            g["corr"], g["min_grip"], g["n_fail_open"], g["n_ep"], c["lang"]["amp"],
            c["lang"]["coh_n"], c["lang"]["floor"]))
    if "box" in c:
        s += " | 框Δ %6.4f 相干n %5.3f" % (c["box"]["amp"], c["box"]["coh_n"])
    return s + "  [%.0fs]" % r["secs"]


# --------------------------------------------------------------------------- 主流程
def main():
    global BATCH
    ap = argparse.ArgumentParser(description="改正后的选点口径: 跨 step 扫描, 三段式")
    ap.add_argument("--run", required=True, choices=list(RUNS) + ["all"])
    ap.add_argument("--steps", type=int, nargs="*", default=None)
    ap.add_argument("--out", default="")
    ap.add_argument("--frames", type=int, default=N_MAE_FRAMES)
    ap.add_argument("--grip-pts", type=int, default=N_GRIP_PTS)
    ap.add_argument("--batch", type=int, default=BATCH)
    args = ap.parse_args()

    BATCH = max(1, args.batch)
    names = list(RUNS) if args.run == "all" else [args.run]
    out = args.out or ("/root/autodl-tmp/val_corrected_%s.json" % args.run)
    allres = {}
    if os.path.exists(out):
        try:
            allres = json.load(open(out))
        except Exception:
            allres = {}

    for name in names:
        cfg = RUNS[name]
        steps = args.steps or list_steps(cfg["ckpt"])
        if not steps:
            print("[%s] ckpt 目录下没有 step: %s" % (name, cfg["ckpt"]))
            continue
        print("\n" + "=" * 110)
        print("[%s] %s   steps=%s" % (name, cfg["ckpt"], steps))
        print("=" * 110)
        eps = load_eps(cfg["frames"])
        print("  val episodes: %d 条" % len(eps), flush=True)
        allres.setdefault(name, {})
        for st in steps:
            t0 = time.time()
            try:
                model = OctoModel.load_pretrained(cfg["ckpt"], step=st)
                r = eval_step(model, cfg, eps, args.frames, args.grip_pts)
                r["secs"] = round(time.time() - t0, 1)
                allres[name][str(st)] = r
                print(fmt(name, st, r), flush=True)
            except Exception as ex:  # noqa: BLE001
                print("  %-8s step %-6d 失败: %s: %s" % (name, st, type(ex).__name__, ex),
                      flush=True)
            json.dump(allres, open(out, "w"), indent=1)

    json.dump(allres, open(out, "w"), indent=1)
    print("\n结果已写入 %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
