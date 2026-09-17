#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""开跑前闸门：新的 target_modules 到底会挂载哪些模块、多少参数。

为什么需要这个：`--peft.target_modules` **匹配失败是静默的**。regex 写错一个字符，
训练照样跑满 20000 步、loss 照样下降，只是什么都没训到。本项目上一版的 regex 里
就躺着三个从未命中的分支（state_proj / action_time_mlp_*），正好是这件事的实证。

三个独立来源交叉核对：
  A. 基座 ckpt 的 safetensors header  -> 候选模块名 + 真实形状
  B. 上一版 adapter 的实际 key         -> 上一版挂了什么（148 张量 / 74 模块）
  C. 新 regex                          -> 新方案会挂什么
判据：C == B + {expert MLP}，一个不多一个不少，且可训练量 == 7,554,048。

⚠️ PEFT 对字符串 target_modules 用的是 re.fullmatch，不是 search —— 这里照抄。

跑法: <smolvla env python> check_target_modules.py
      WORK=/data/toy <python> check_target_modules.py
"""
import glob
import json
import os
import re
import struct
import sys
from collections import defaultdict

WORK = os.environ.get("WORK", "/root/autodl-tmp")
RANK = 16

BASE = os.path.join(WORK, "pi05_base")
PREV = os.path.join(WORK, "outputs", "pi05_lora_r16", "checkpoints",
                    "018000", "pretrained_model", "adapter_model.safetensors")

# 上一版脚本里的 regex（原样抄）
OLD = (r'.*\.paligemma\..*\.language_model\..*\.self_attn\.(q_proj|v_proj)'
       r'|.*\.gemma_expert\..*\.self_attn\.(q_proj|v_proj)'
       r'|model\.(state_proj|action_in_proj|action_out_proj'
       r'|action_time_mlp_in|action_time_mlp_out)')

# 本版：上一版 + expert MLP gate/up/down，并删掉三个从未命中的死分支
NEW = (r'.*\.paligemma\..*\.language_model\..*\.self_attn\.(q_proj|v_proj)'
       r'|.*\.gemma_expert\..*\.(self_attn\.(q_proj|v_proj)'
       r'|mlp\.(gate_proj|up_proj|down_proj))'
       r'|model\.(action_in_proj|action_out_proj)')


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def base_modules():
    """基座里所有 2D Linear 权重的 (模块名 -> [out, in])。

    ⚠️ 基座 ckpt 的 key 是**平铺的**（`action_in_proj.weight`、
    `paligemma_with_expert...`），而真实 nn.Module 路径挂在 `PI05Policy.model`
    下面（`model.action_in_proj`）。所以这里补上 `model.` 前缀，否则拿平铺名
    去和 adapter 的真实路径对拍会整体错位（踩过：报了一堆假的 missing/extra，
    总数还正好少 33,792 = 两个动作投影）。
    证据：adapter key = `base_model.model` + `model.paligemma_with_expert...`
    """
    files = sorted(glob.glob(os.path.join(BASE, "*.safetensors")))
    mods = {}
    for fp in files:
        h = read_header(fp)
        for k, v in h.items():
            if k == "__metadata__":
                continue
            if not k.endswith(".weight") or len(v["shape"]) != 2:
                continue
            ms = k[:-len(".weight")]
            if not ms.startswith("model."):
                ms = "model." + ms
            mods[ms] = v["shape"]
    return mods, files


def adapter_modules(path):
    """adapter 里实际挂载的模块名集合 + 张量数。"""
    h = read_header(path)
    keys = [k for k in h if k != "__metadata__"]
    mods = set()
    for k in keys:
        for suf in (".lora_A.weight", ".lora_B.weight"):
            if k.endswith(suf):
                m = k[:-len(suf)]
                if m.startswith("base_model.model."):
                    m = m[len("base_model.model."):]
                mods.add(m)
    return mods, len(keys)


def lora_params(shape):
    """nn.Linear weight [out, in] 挂 r 阶 LoRA 的参数量。"""
    out, inn = shape
    return RANK * inn + out * RANK


def group(m):
    if "gemma_expert" in m and ".mlp." in m:
        return "expert MLP (新增)"
    if "gemma_expert" in m:
        return "expert attn q/v (上一版)"
    if "language_model" in m:
        return "VLM attn q/v (上一版)"
    return "action proj (上一版)"


def main():
    if not os.path.isdir(BASE):
        print(f"!! 找不到基座目录: {BASE}（用 WORK=... 指定）")
        return 1

    mods, files = base_modules()
    print(f"[基座] {BASE}: {len(files)} 个 safetensors, "
          f"{len(mods)} 个 2D Linear 模块")

    hit_new = {m: s for m, s in mods.items() if re.fullmatch(NEW, m)}
    hit_old = {m: s for m, s in mods.items() if re.fullmatch(OLD, m)}
    print(f"[regex] OLD 命中 {len(hit_old)} 个模块 / "
          f"NEW 命中 {len(hit_new)} 个模块")

    tot_all = sum(lora_params(s) for s in hit_new.values())
    agg = defaultdict(lambda: [0, 0])
    for m, s in hit_new.items():
        g = group(m)
        agg[g][0] += lora_params(s)
        agg[g][1] += 1

    print(f"\n{'=' * 78}\n新 regex 命中的模块（按组）\n{'=' * 78}")
    for g, (p, c) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        print(f"  {g:<26s}{p:>12,}{100 * p / tot_all:>8.1f}%{c:>6d} 个模块")
    print(f"  {'-' * 60}")
    tot_new = sum(v[0] for v in agg.values())
    print(f"  {'合计':<26s}{tot_new:>12,}   ({len(hit_new)} 个模块)")

    tot_old = sum(lora_params(s) for s in hit_old.values())
    print(f"\n  上一版 regex 理论上应挂: {tot_old:,} ({len(hit_old)} 个模块)")

    # ---- 与上一版 adapter 实测对拍（adapter 不在时只报理论值）----
    n_extra, missing, same = -1, set(), False
    if os.path.exists(PREV):
        print(f"\n{'=' * 78}\n与上一版 adapter 实测对拍\n{'=' * 78}")
        amd, nk = adapter_modules(PREV)
        print(f"[adapter] {nk} 个张量, {len(amd)} 个模块")
        same = set(hit_old) == amd
        print(f"  regex(OLD) 模块集 == adapter 实际模块集 ? {same}")
        if not same:
            print(f"    regex 有但 adapter 没有: {sorted(set(hit_old) - amd)}")
            print(f"    adapter 有但 regex 没有: {sorted(amd - set(hit_old))}")

        extra = set(hit_new) - amd
        missing = amd - set(hit_new)
        n_extra = len(extra)
        print(f"\n  NEW − 上一版(实测) = {n_extra} 个（应全部是 expert MLP）:")
        ex_agg = defaultdict(int)
        for m in extra:
            ex_agg[group(m)] += 1
        for g, c in ex_agg.items():
            print(f"    {c:>3d} 个  {g}")
        for m in sorted(extra)[:3] + (["  ..."] if len(extra) > 6 else []) \
                + sorted(extra)[-3:]:
            if m == "  ...":
                print(m)
            else:
                print(f"      e.g. {m}  {hit_new[m]} -> {lora_params(hit_new[m]):,}")
        print(f"  上一版 − NEW = {len(missing)} 个（应为 0）: {sorted(missing)}")
    else:
        print(f"\n  (跳过对拍：找不到上一版 adapter {PREV})")
        print("  ⇒ 只做理论值核对，下面按总量判据。")

    # 顺带证明：上一版 regex 里那三个分支确实是死的（从未命中任何模块）
    for dead in ("state_proj", "action_time_mlp_in", "action_time_mlp_out"):
        n = sum(1 for m in mods if m.endswith("." + dead))
        print(f"  [死分支核查] 基座里 `{dead}` 模块数 = {n}"
              f"{'  ← 确认不存在' if n == 0 else '  ← ⚠️ 存在！'}")

    print(f"\n{'=' * 78}")
    print(f"  ⇒ 预期可训练参数量 = {tot_new:,}")
    print(f"  ⇒ 相对上一版 {tot_old:,} 增量 = +{tot_new - tot_old:,}")
    print(f"{'=' * 78}")

    # 判据：模块总数 128；有新 adapter 时再加「增量恰为 54、且无缺失」两条
    ok = len(hit_new) == 128 and tot_new == 7554048
    if n_extra >= 0:
        ok = ok and same and not missing and n_extra == 54
    print(f"\n闸门判定: {'PASS' if ok else 'FAIL'}"
          f"   (期望 128 个模块 / 7,554,048 参数 / 新增 54 个 = 18 层 x 3)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
