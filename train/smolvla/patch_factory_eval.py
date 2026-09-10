# -*- coding: utf-8 -*-
"""把 lerobot `make_train_eval_datasets` 的 eval 划分从「每组末尾 N 条」改成「整组均匀抽样」。

问题：
    lerobot 默认按任务分组，取每组的**末尾** ceil(n * eval_split) 条做验证集。
    本项目三类玩具虽然标签不同，但若训练成单任务标签、且数据是**按玩具分块连续采集**的，
    那么「末尾 N 条」就等于「最后一段的那个玩具」—— 验证集完全落在一类玩具上，
    验证 loss 也就无法反映模型在另外两类上的表现。

改法：
    把每组的验证集索引按步长均匀铺开（`{0, step, 2*step, ...}`），
    抽出的 N 条必然散布在整个 episode 集合上，从而跨过所有采集分段。
    不依赖任何玩具标签，只假设数据是分块采集的。

用法：
    LEROBOT_SRC=/root/autodl-tmp/lerobot python patch_factory_eval.py

    135 条 × eval_split=0.11 → 15 条验证，抽出的是 {0, 9, 18, ..., 126}，
    恰好覆盖三个玩具分段（0-44 / 45-89 / 90-134）。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

DEFAULT_SRC = "/root/autodl-tmp/lerobot"

OLD = """    train_episodes, eval_episodes = [], []
    for eps in task_to_episodes.values():
        n_eval = math.ceil(len(eps) * cfg.dataset.eval_split)
        train_episodes.extend(eps[: len(eps) - n_eval])
        eval_episodes.extend(eps[len(eps) - n_eval :])"""

NEW = """    train_episodes, eval_episodes = [], []
    for eps in task_to_episodes.values():
        n_eval = math.ceil(len(eps) * cfg.dataset.eval_split)
        n_eval = min(n_eval, len(eps) - 1)  # 至少留 1 条给训练
        if n_eval <= 0:
            train_episodes.extend(eps)
            continue
        # Stratified sampling: spread the eval picks EVENLY across each task group. For a single
        # task recorded in contiguous phases (e.g. 3 toys in blocks), taking the group's LAST
        # n_eval episodes would hold out only one phase/toy. Uniform picks span the whole group,
        # so the held-out set covers every phase/toy. Keeps multi-task behaviour sane too.
        step = len(eps) / n_eval
        eval_set = {int(k * step) for k in range(n_eval)}
        i = 0
        while len(eval_set) < n_eval and i < len(eps):
            eval_set.add(i)
            i += 1
        train_episodes.extend([eps[i] for i in range(len(eps)) if i not in eval_set])
        eval_episodes.extend([eps[i] for i in sorted(eval_set)])"""


def main() -> int:
    root = Path(os.environ.get("LEROBOT_SRC", DEFAULT_SRC))
    target = root / "src" / "lerobot" / "datasets" / "factory.py"
    if not target.is_file():
        print(f"FAIL: 找不到 {target}（用 LEROBOT_SRC 指定 lerobot checkout 根目录）", file=sys.stderr)
        return 1

    s = target.read_text(encoding="utf-8")
    if OLD not in s:
        print("FAIL: 没找到待替换的原始代码块（可能已打过补丁，或 lerobot 版本漂移）", file=sys.stderr)
        return 1
    if NEW in s:
        print("SKIP: 已打过补丁")
        return 0

    target.write_text(s.replace(OLD, NEW), encoding="utf-8")
    print(f"OK: 已打补丁 -> {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
