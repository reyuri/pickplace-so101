# -*- coding: utf-8 -*-
# SO-101 从臂双相机数据采集(真机示教采集)
# 用 Python 配置对象直调 record(),绕开 Windows 命令行引号地狱
#
# 用法:
#   conda activate lerobot
#   python record_so101.py
#
# 3 个玩偶, 每个一个独立数据集文件夹, 每个最多 NUM_EPISODES 条:
#   小象(elephant) / 树(tree) / 球(ball)
# 每次只采一个玩偶; 采完换 DOLL 变量重跑一次(共跑 3 次)。
#
# 操作(监听全局按键,不用点终端窗口):
#   → 右箭头  提前结束当前一条采集
#   ← 左箭头  放弃当前条并重录
#   Esc       停止全部采集
# 两条之间是 reset 阶段(不记录),手动把主臂搬回原位 + 重置道具,按 → 跳过或等 reset_time_s 到点。
'''
每采完一个 toy,python verify_dataset.py → 自检数据
'''

from pathlib import Path

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.scripts.lerobot_record import DatasetRecordConfig, RecordConfig, record
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig

# ================= 需要你改的地方 =================
SINGLE_TASK = "Pick up the tree and place it into the box"   # 训练指令。三个玩偶用同一句,单任务泛化。
FPS = 30
W, H = 640, 480
WARMUP_S = 5     # connect 时多读 5s、丢弃冷启动帧(自动曝光/对焦 settle),稳过 warmup→record 交接处 200ms 超时
MAX_PER_DOLL = 45   # 每个玩偶最多采集条数
# 相机配置里的 fps 必须给(SOFollowerRobotConfig 校验要求),但实际上 lerobot 的
# `_configure_capture_settings` 已被改写成【纯 set、零回读、不设 FPS】——这个 fps 值只用于
# 数据集/显示层,不会再触发 CAP_PROP_FPS 的 set/get(那才是双开会坏 1fps 的元凶)。见 camera_opencv.py。

# 数据集根目录(项目根已切到 VLA_Yolo)
BASE = Path("D:/My_Work/202510-findAremote/Projects/VLA/VLA_Yolo/data/reyqiao")

# 三个玩偶: 键 -> (中文标签, 数据集文件夹名)。采集哪个改 DOLL 即可。
DOLLS = {
    "elephant": ("小象", "lerobot_rey_toy_elephant"),
    "tree":     ("树",   "lerobot_rey_toy_tree"),
    "ball":     ("球",   "lerobot_rey_toy_ball"),
}
# 本次采集的玩偶: "elephant" / "tree" / "ball"(跑完换一个再跑)
DOLL = "tree"
# =================================================


def build_config(doll_name: str, num_episodes: int):
    cn, folder = DOLLS[doll_name]
    # ---- 从臂(被控,采集对象), 双相机 ----
    # side        = 相机 index 1(外接 USB/Brio); 640x480@30 是这颗能稳定跑的档(720p/1080p 只有~5fps, 会录崩)
    # eye_in_hand = 相机 index 2
    robot = SOFollowerRobotConfig(
        port="COM3",
        id="zihao_follower_arm",
        cameras={
            "side": OpenCVCameraConfig(
                index_or_path=1, width=W, height=H, fps=FPS, fourcc="MJPG", warmup_s=WARMUP_S,
            ),
            "eye_in_hand": OpenCVCameraConfig(
                index_or_path=2, width=W, height=H, fps=FPS, fourcc="MJPG", warmup_s=WARMUP_S,
            ),
        },
    )

    # ---- 主臂(遥操作) ----
    teleop = SOLeaderTeleopConfig(port="COM4", id="zihao_leader_arm")

    # ---- 数据集(每个玩偶独立文件夹) ----
    dataset = DatasetRecordConfig(
        repo_id=f"reyqiao/{folder}",        # 上传 HF 时用的名字
        single_task=SINGLE_TASK,
        root=str(BASE / folder),            # root 就是数据集目录本身(不是父目录)。重新采集先删掉该目录。
        fps=FPS,                            # 数据集帧率(录制循环按 30fps 存帧, 相机 60fps 只取最新帧)
        episode_time_s=30,                  # 每条最长 30 秒(上限, 不是目标; 按 → 可提前结束)
        reset_time_s=10,                    # 两条之间重置场景窗口(搬主臂回位不够)
        reset_min_time_s=4,                 # reset 前 4 秒内按 → 无效(丢弃), 防止录制尾部连发 → 跳过 reset
        num_episodes=num_episodes,
        video=True,
        push_to_hub=False,                  # 本地保存, 不上传(想采完自动传 HF 就改 True, 走镜像)
        vcodec="h264",                      # 比默认 AV1 编码快
        # 批编码 =5(采完 5 条最后一次性编码)。原版 lerobot 0.4.4 的批编码在 3 处都会崩(meta.episodes 为
        # None、episodes parquet 未 finalize、同路径重开截断), 已 patch lerobot_dataset.py 的
        # `_batch_save_episode_video`:批触发时 flush→close→滚动 chunk/file 索引→重载索引。
        video_encoding_batch_size=5,
    )

    return RecordConfig(
        robot=robot,
        teleop=teleop,
        dataset=dataset,
        display_data=True,  # 打开 rerun 窗口看双相机 + 关节轨迹
        play_sounds=True,   # 语音播报 "Recording episode N" / "Reset the environment",当采集提示音用
        resume=True,        # 续采:加载已有数据集目录。想从零重采,改回 resume=False 并删目录。
    )


if __name__ == "__main__":
    cn, folder = DOLLS[DOLL]
    print(f"==== 采集玩偶: {cn}({DOLL}) / 数据集 {folder} / 最多 {MAX_PER_DOLL} 条 ====")
    print("文件夹:", BASE / folder)
    cfg = build_config(DOLL, MAX_PER_DOLL)
    record(cfg)
