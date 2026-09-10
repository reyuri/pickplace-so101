#!/usr/bin/env python3
"""SmolVLA 通用训练启动器:按数据集实际相机数自动生成 policy.input_features + empty_cameras。

用法:
    python smolvla_train_launcher.py --dataset-root=/root/autodl-tmp/lerobot_rey_toy --steps=40000 [其它 lerobot-train 参数]

自动处理:
  - 扫描 dataset meta/info.json 的 features,数出 observation.images.* 相机键数 K
  - input_features = 每个真实相机键 -> {type:VISUAL, shape:[3,256,256]}
  - empty_cameras = 3 - K(SmolVLA 视觉塔固定 3 槽,空槽用 -1 dummy 填充)

单相机(front)   -> K=1, input={front}, empty=2
双相机(front+hand)-> K=2, input={front,hand}, empty=1
三相机            -> K=3, input={3 键}, empty=0

然后把参数拼进 lerobot-train 命令行执行。
"""
import json
import subprocess
import sys
from pathlib import Path

# 固定参数(单/双相机通用,不需要改)
BASE_ARGS = [
    "--policy.path=lerobot/smolvla_base",
    "--policy.push_to_hub=false",
    "--policy.repo_id=reyqiao/lerobot_rey_toy",
    "--dataset.repo_id=reyqiao/lerobot_rey_toy",
    "--dataset.streaming=false",
    "--dataset.eval_split=0.1",
    "--output_dir=/root/autodl-tmp/outputs/train",
    "--job_name=rey_toy_smolvla",
    "--policy.device=cuda",
    "--wandb.enable=false",
    # 默认训练超参(会被命令行覆写)
    "--steps=40000",
    "--batch_size=8",
    "--num_workers=4",
    "--eval_steps=2000",
    "--save_freq=2000",
]


def find_dataset_root(argv: list[str]) -> Path:
    for a in argv:
        if a.startswith("--dataset.root="):
            return Path(a.split("=", 1)[1])
        if a.startswith("--dataset-root="):
            return Path(a.split("=", 1)[1])
    return Path("/root/autodl-tmp/lerobot_rey_toy")


def detect_cameras(dataset_root: Path) -> list[str]:
    """从 meta/info.json 的 features 找出所有 observation.images.* 相机键,按出现顺序返回。"""
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    features = info["features"]
    cameras = [k for k in features if k.startswith("observation.images.")]
    # 过滤掉 empty 占位(如果有)
    cameras = [c for c in cameras if not c.startswith("observation.images.empty")]
    if not cameras:
        print(f"[launcher] WARNING: 数据集 {dataset_root} 没有任何 observation.images.* 相机!", file=sys.stderr)
    return cameras


def build_args(cameras: list[str], user_args: list[str]) -> list[str]:
    K = len(cameras)
    empty = 3 - K
    if empty < 0:
        print(f"[launcher] 相机数 {K} > 3,超过 SmolVLA 视觉塔容量,请检查数据集!", file=sys.stderr)
        empty = 0
    input_features = {c: {"type": "VISUAL", "shape": [3, 256, 256]} for c in cameras}
    out = [
        f"--policy.input_features={json.dumps(input_features, separators=(',', ':'))}",
        f"--policy.empty_cameras={empty}",
    ]
    # BASE_ARGS(可被 user_args 覆盖,后面的优先)
    overridable = {k.split("=", 1)[0] for k in BASE_ARGS}
    for a in BASE_ARGS:
        key = a.split("=", 1)[0]
        if key in {u.split("=", 1)[0] for u in user_args}:
            continue  # 用户显式传了,交给用户参数
        out.append(a)
    out.extend(user_args)
    return out


def main() -> None:
    argv = sys.argv[1:]
    dataset_root = find_dataset_root(argv)
    cameras = detect_cameras(dataset_root)
    empty = 3 - len(cameras)
    print(f"[launcher] 检测到 {len(cameras)} 个相机:", cameras, "| empty_cameras =", empty, flush=True)
    args = build_args(cameras, argv)
    print("[launcher] 完整命令:", flush=True)
    print("  lerobot-train " + " ".join(args), flush=True)
    print("[launcher] 开始执行 (见下方输出)...", flush=True)
    sys.exit(subprocess.run(["lerobot-train", *args]).returncode)


if __name__ == "__main__":
    main()
