#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SO-101 双相机闭环驱动器(云端推理对比 A1/A2/A3):本地读 侧视+手眼 + 状态 → 云端 /infer → 动作。

架构:
  云端(服务器 4090)每模型一个 serve(A1:6010 纯推理 / A2:6011、A3:6012 带红框注入)。
  serve 收双图 {side, eye_in_hand} + state + task,返回 {action:[50,6], box:[x1,y1,x2,y2]|null}。
  A2/A3 的 serve 会用 YOLO 对 side 画红框后喂模型(与训练 bake 一致);A1 侧视原图进模型。

双相机(与 record_so101.py 一致):
  side=index1, eye_in_hand=index2, 均 640x480@30 MJPG。
  注意: SOFollower 相机 color_mode=RGB, 转 BGR 存/编码; 红框坐标在整帧像素系, 显示侧即画侧。

实时显示(任务重点: side 窗口 = 真实进入模型的**带红框**图):
  后台线程 LiveDisplay 用共享缓冲(control 线程每次 inference 后写入最新 side_bgr/eih_bgr + 服务端返回的 box),
  把红框画到 side 帧再 imshow → **显示内容 = 送进模型的那张**。显示线程**不碰机器人**(无串口/相机访问),
  只对最新帧做 cv2.rectangle(微秒级) → 绝不拖慢 30Hz 控制回路(逐步下发/对模型目标是独立路径)。

用法(本地 Windows,臂=COM3,云端 serve 已起 + SSH 隧道已通):
  # 先只读核验(不动臂): 取实时双相机+状态, 推理, 显示 side(带框)+eye_in_hand, 打印动作
  python drive_so101_dual.py --model A2 --toy tree --read-only
  # 真机动臂闭环
  python drive_so101_dual.py --model A2 --toy tree --move --max-cycles 5

安全: --abs-clamp 绝对钳制 / --diverge-stop 发散即停 / --move 前应先 --read-only; 收尾回 --home-state。
"""
import argparse
import base64
import os
import threading
import time
import logging
from datetime import datetime

# 关掉机器人臂库"Relative goal position magnitude had to be clamped"刷屏 WARNING
# (Octo 动作大、每 cycle 相对目标超 max_rel → 每步重复警告,淹没关键 print)。
# drive 关键输出用 print()(不受 logging 影响)。双保险:
#   ① root 级别抬到 ERROR(WARNING 连 record 都不生成,最稳)
#   ② 再加只滤这条消息的 Filter(万一有别的代码擅自把级别调回,仍能精确掐掉它)
# 其它真实 WARNING/ERROR(校准、通信异常)仍为 ERROR 可见,不误伤。
class _ClampWarningFilter(logging.Filter):
    def filter(self, record):
        return "Relative goal position magnitude had to be clamped" not in record.getMessage()

logging.getLogger().setLevel(logging.ERROR)
logging.getLogger().addFilter(_ClampWarningFilter())

import numpy as np
import cv2

import drive_so101 as cli  # 复用: MOTOR_ORDER / SAFE_ABS_RANGE / encode_bgr_jpeg / clamp_action / velocity_limit / drive_to / execute_smooth / parse_num

RED = (0, 0, 255)  # BGR, 与 bake_redbox_dataset.py 一致
BOX_THICK = 3
DISPLAY_SCALE = 0.7  # 拼接窗缩放: 640x960 -> ~448x672(用户要求更小便于观察)

# --model -> (url, 是否红框模型, 默认 task 前缀)。红框模型(A2/A3)无需本地判断, 服务端会返回 box。
URLS = {"A1": "http://127.0.0.1:6010", "A2": "http://127.0.0.1:6011", "A3": "http://127.0.0.1:6012",
        "OCTO": "http://127.0.0.1:6020"}
BOXED = {"A1": False, "A2": True, "A3": True, "OCTO": False}  # A1 训练无框→侧视原图; A2/A3 训练有框→服务端注入; OCTO 无框
TOYS = ("elephant", "tree", "ball")

# 起位/收尾 home: 取自训练数据集 `VLA_Proj/data_v1/reyqiao/lerobot_rey_toy` 各 episode 起始 obs.state 的
# 中位数(2026-09-07 统计 75 个 ep 一次写死, 不再每次启动扫描数据集)。保证进出都在分布内姿态(下垂静止位)。
DEFAULT_HOME = "-0.775,-98.827,99.369,75.314,39.145,0.544"

# 位姿+夹爪轨迹 任务完成检测的总开关: 想默认开启就把这里改 True, 否则每次命令行加 --success-detect。
# 命令行可随时 --no-success-detect 强制关掉(覆盖此开关)。
SUCCESS_DETECT = False


def task_for(model, toy):
    if model == "A3":
        return f"Pick up the {toy} in red box and place it into the box"
    return f"Pick up the {toy} and place it into the box"


# --------------------------------------------------------------------------- 相机/机器人
def make_robot_config_dual(port, robot_id, max_rel, cam_side=1, cam_eye=0):
    """双相机配置: side, eye_in_hand。

    ⚠️ 索引默认值 side=1 / eye_in_hand=0 是 2026-09-10 实测修正的(原写死 side=1/eye=2)。
    Windows DSHOW 的设备枚举顺序**会漂移**, 与插拔顺序无关(内置摄像头拔不掉, 会永久占位)。
    实测现状: idx0=`USB Camera`(手眼, 俯视夹爪) / idx1=`Brio 90`(侧视全景) / idx2=笔记本内置。

    跑真机前先 `python dual_camera_viewer.py 1 0` 目视确认这两路画面;
    若又漂了, 用 `--cam-side/--cam-eye` 覆盖即可, 不必改代码。
    """
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    def cam(idx):
        return OpenCVCameraConfig(index_or_path=idx, width=640, height=480, fps=30, fourcc="MJPG")

    return SOFollowerRobotConfig(
        port=port, id=robot_id, max_relative_target=max_rel,
        cameras={"side": cam(cam_side), "eye_in_hand": cam(cam_eye)},
    )


def make_robot_dual(port, robot_id, max_rel, cam_side=1, cam_eye=0):
    from lerobot.robots.utils import make_robot_from_config
    return make_robot_from_config(
        make_robot_config_dual(port, robot_id, max_rel, cam_side, cam_eye))


def read_live_obs_dual(robot, save_prefix="live_dual"):
    """读实时状态 + 双相机。返回 (state6, side_bgr, eih_bgr)。"""
    import cv2

    obs = robot.get_observation()  # RGB (H,W,3)
    state = [float(obs[f"{m}.pos"]) for m in cli.MOTOR_ORDER]
    side_bgr = cv2.cvtColor(obs["side"], cv2.COLOR_RGB2BGR)
    eih_bgr = cv2.cvtColor(obs["eye_in_hand"], cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{save_prefix}_side.jpg", side_bgr)
    cv2.imwrite(f"{save_prefix}_eye.jpg", eih_bgr)
    return state, side_bgr, eih_bgr


def read_state(robot):
    """仅读机器人关节(总线, M100), 不读相机。等价 get_observation 的姿态部分(同是 Present_Position)。"""
    cur = robot.bus.sync_read("Present_Position")
    return [float(cur[m]) for m in cli.MOTOR_ORDER]


def infer_dual(url, side_bgr, eih_bgr, state, task, timeout=120.0):
    """双图 POST /infer。返回 resp(dict, 含 action/box/infer_ms)。"""
    import requests

    def b64(bgr):
        import cv2
        ok, buf = cv2.imencode(".jpg", bgr)
        if not ok:
            raise RuntimeError("JPEG 编码失败")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    payload = {"side": b64(side_bgr), "eye_in_hand": b64(eih_bgr),
               "state": [float(x) for x in state], "task": task}
    resp = requests.post(url.rstrip("/") + "/infer", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def draw_box(bgr, box, toy=None):
    """把服务端返回的框(整帧像素系)画到 bgr(原地), 供显示。box=[x1,y1,x2,y2]。"""
    import cv2
    if not box or len(box) != 4:
        return
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(bgr, (x1, y1), (x2, y2), RED, BOX_THICK)
    if toy:
        cv2.putText(bgr, str(toy), (x1, max(y1 - 6, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)


# --------------------------------------------------------------------------- 双相机拼接 + 录像(默认关, 不碰机器人/相机)
def stack_dual(side_bgr, eih_bgr, box=None, toy=None):
    """side 上 / eye_in_hand 下, 各 640x480 → 拼接为 640x960 一图(布局同实时显示)。"""
    import cv2

    def tile(img, label):
        t = cv2.resize(img.copy(), (640, 480))
        cv2.rectangle(t, (0, 0), (639, 22), (40, 40, 40), -1)
        cv2.putText(t, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return t

    sd = tile(side_bgr, "SIDE")
    draw_box(sd, box, toy)
    ed = tile(eih_bgr, "EYE_IN_HAND")
    return np.vstack((sd, ed))


class VidRecorder:
    """把拼接的双目帧写入视频(惰性打开 VideoWriter)。只收 frame, 不做任何机器人/相机访问。"""

    def __init__(self, path, fps):
        self.path, self.fps = path, fps
        self.w = None

    def write(self, frame):
        import cv2
        if self.w is None:
            self.w = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"MJPG"), self.fps,
                                     (frame.shape[1], frame.shape[0]))
            if not self.w.isOpened():
                print(f"[record] 打开视频失败: {self.path}")
                self.w = None
                return
            print(f"[record] 开始录制: {self.path}  ({frame.shape[1]}x{frame.shape[0]}, {self.fps}fps)")
        self.w.write(frame)

    def release(self):
        if self.w is not None:
            self.w.release()
            print(f"[record] 已保存: {self.path}")
            self.w = None


# 项目根/outputs: 录制视频默认落盘目录(项目根=本脚本所在目录 = VLA_Yolo)。
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def make_recorder(args):
    """record-video 开则返回 VidRecorder, 否则 None(默认关=零开销)。
    默认保存路径 = 项目根/outputs/dual_<model>_<toy>_<时间戳>.avi(自动 mkdir); --record-path 显式给则优先。"""
    if not args.record_video:
        return None
    path = args.record_path or os.path.join(
        OUTPUT_DIR, f"dual_{args.model}_{args.toy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return VidRecorder(path, args.record_fps)


# --------------------------------------------------------------------------- 相机流缓冲 + 抓帧线程 + 显示线程
class CamStream:
    """共享相机/框缓冲: 抓帧线程写 frames, 控制线程写 box, 显示/录制线程读。线程安全。
    相机句柄只由一条抓帧线程读(避免 DSHOW 双读饿死); 不碰机器人总线。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.side = None
        self.eih = None
        self.box = None
        self.frames = 0
        self._first = threading.Event()

    def set_frames(self, side, eih):
        with self.lock:
            self.side, self.eih = side, eih
            self.frames += 1
        self._first.set()

    def set_box(self, box):
        with self.lock:
            self.box = box

    def set(self, side, eih, box):
        with self.lock:
            self.side, self.eih, self.box = side, eih, box
            self.frames += 1
        self._first.set()

    def latest(self):
        with self.lock:
            return self.side, self.eih, self.box

    def wait_first(self, timeout=5.0):
        self._first.wait(timeout)


class GrabThread(threading.Thread):
    """专属相机抓帧线程: 顺序读 side+eye(同 get_observation 的串行 read), 写 CamStream,
    供显示/录制/推理取最新帧(≈30fps 平滑流)。只碰相机 USB, 绝不碰机器人总线(与控制线程零冲突)。"""

    def __init__(self, robot, stream, toy=None, recorder=None, target_fps=30.0, record_box=False):
        super().__init__(daemon=True)
        self.cam_side = robot.cameras["side"]
        self.cam_eih = robot.cameras["eye_in_hand"]
        self.stream = stream
        self.toy = toy
        self.recorder = recorder
        self.target_fps = max(1.0, float(target_fps))
        self.record_box = record_box  # 默认关: 录制画纯净实时流(框只在算法侧注入)
        self._run = True
        self.measured_fps = 0.0

    def stop(self):
        self._run = False

    def run(self):
        import cv2
        period = 1.0 / self.target_fps
        t_prev = time.perf_counter()
        n = 0
        t0 = time.perf_counter()
        while self._run:
            try:
                side = self.cam_side.async_read()   # RGB(get_observation 同款)
                eih = self.cam_eih.async_read()      # RGB
                side_bgr = cv2.cvtColor(side, cv2.COLOR_RGB2BGR)
                eih_bgr = cv2.cvtColor(eih, cv2.COLOR_RGB2BGR)
                self.stream.set_frames(side_bgr, eih_bgr)
                if self.recorder is not None:
                    # 默认不画框(录纯净实时流): 框只在其对应的决策帧有语义, 画到新帧会滞后于物体。
                    box = self.stream.latest()[2] if self.record_box else None
                    self.recorder.write(stack_dual(side_bgr, eih_bgr, box, self.toy))
                n += 1
                now = time.perf_counter()
                if now - t0 >= 2.0:
                    self.measured_fps = n / (now - t0)
                    n, t0 = 0, now
            except Exception as e:  # noqa: BLE001
                print(f"[grab] 读相机失败: {type(e).__name__}: {e}")
                time.sleep(0.05)
            dt = time.perf_counter() - t_prev
            if dt < period:
                time.sleep(period - dt)
            t_prev = time.perf_counter()


class LiveDisplay(threading.Thread):
    """后台显示: 每~50fps 从共享 CamStream 取最新 side/eih/box, 画框(客户端画服务端返回的 box) imshow。
    只读缓冲线程, 不碰机器人/相机, 不拖慢控制。"""

    def __init__(self, stream, toy=None):
        super().__init__(daemon=True)
        self.stream = stream
        self.toy = toy
        self._run = True

    def stop(self):
        self._run = False

    def run(self):
        import cv2
        while self._run:
            s, e, b = self.stream.latest()
            tiles = []
            if s is not None:
                sd = s.copy()
                draw_box(sd, b, self.toy)
                sd = cv2.resize(sd, (640, 480))
                cv2.rectangle(sd, (0, 0), (639, 22), (40, 40, 40), -1)
                cv2.putText(sd, "SIDE", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                tiles.append(sd)
            if e is not None:
                ed = e.copy()
                ed = cv2.resize(ed, (640, 480))
                cv2.rectangle(ed, (0, 0), (639, 22), (40, 40, 40), -1)
                cv2.putText(ed, "EYE_IN_HAND", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                tiles.append(ed)
            if tiles:
                v = np.vstack(tiles) if len(tiles) > 1 else tiles[0]
                cv2.imshow("dual_cam (side+eye)", cv2.resize(v, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE))
            k = cv2.waitKey(20) & 0xFF  # ~50fps 重绘, 只重blit共享帧, 不阻塞
            if k == ord("q"):
                self._run = False
                break
        cv2.destroyAllWindows()


# --------------------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description="SO-101 双相机闭环(云端推理 A1/A2/A3)")
    ap.add_argument("--model", choices=list(URLS.keys()), required=True, help="A1/A2/A3")
    ap.add_argument("--toy", choices=TOYS, default="tree", help="目标玩具(默认任务沿用; --task 覆盖)")
    ap.add_argument("--task", default="", help="任务文本; 空则按 model+toy 自动拼")
    ap.add_argument("--url", default="", help="云端 /infer 地址; 空则按 --model 映射(A1:6010/A2:6011/A3:6012)")
    ap.add_argument("--read-only", action="store_true", help="只读: 取帧+推理+显示, 绝不动臂")
    ap.add_argument("--move", action="store_true", help="真机闭环(receding horizon)")
    ap.add_argument("--display", action="store_true", default=False, help="开启实时显示(默认关, 避免无GUI环境卡顿)")
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--robot-id", default="zihao_follower_arm")
    ap.add_argument("--cam-side", type=int, default=1,
                    help="侧视相机索引(默认1=Brio 90); 枚举漂移时用 dual_camera_viewer.py 目视确认后覆盖")
    ap.add_argument("--cam-eye", type=int, default=0,
                    help="手眼(eye_in_hand)相机索引(默认0=USB Camera); 枚举漂移时覆盖, 不必改代码")
    ap.add_argument("--max-rel", type=float, default=10.0, help="每步相对目标上限(安全)")
    # 闭环控制参数
    ap.add_argument("--max-cycles", type=int, default=20)
    ap.add_argument("--exec-steps", type=int, default=15)
    ap.add_argument("--smooth", action="store_true")
    ap.add_argument("--smooth-margin", type=float, default=1.0)
    ap.add_argument("--max-sub", type=int, default=10)
    ap.add_argument("--vel-limit", type=float, default=0.0)
    ap.add_argument("--abs-clamp", action="store_true", default=True)
    ap.add_argument("--diverge-stop", action="store_true")
    ap.add_argument("--diverge-margin", type=float, default=5.0)
    # 双目串台守卫(默认开): 2026-09-10 真机闭环第7轮起手眼(idx0)画面被换成侧视(idx1)画面,
    # 模型吃到错图却没有任何告警, 整跑结论作废。此后每轮比对两路画面, 异常即落盘+中止。
    ap.add_argument("--no-swap-guard", action="store_false", dest="swap_guard",
                    help="关闭双目串台守卫(默认开)")
    ap.add_argument("--swap-thresh", type=float, default=15.0,
                    help="两路画面平均绝对差低于此判为串台; 实测正常 63~65(cam_watch.py), 默认15留足余量")
    ap.add_argument("--stale-warn", type=int, default=2,
                    help="连续这么多轮推理输入逐像素完全相同即告警(抓帧线程停滞), 默认2")
    # 任务完成检测(位姿+夹爪轨迹门控; 默认关, 开则命中立刻回 home)
    ap.add_argument("--success-detect", action="store_true", default=SUCCESS_DETECT,
                    help="位姿+夹爪轨迹判断任务完成, 命中立刻回 home(默认由顶部 SUCCESS_DETECT 开关决定)")
    ap.add_argument("--no-success-detect", action="store_false", dest="success_detect",
                    help="强制关闭任务完成检测(覆盖顶部开关)")
    ap.add_argument("--done-grip-open", type=float, default=30.0, help="夹爪开阈值(>= 判张开)")
    ap.add_argument("--done-grip-close", type=float, default=10.0, help="夹爪闭阈值(<= 判'抓过一次')")
    ap.add_argument("--done-lift-floor", type=float, default=-50.0, help="成功需 shoulder_lift>=此(排除下垂home)")
    ap.add_argument("--done-elbow-ceil", type=float, default=0.0, help="成功需 elbow_flex<=此(排除低位套玩具)")
    ap.add_argument("--done-sustain", type=int, default=1, help="连续命中成功条件的轮数(防瞬态)")
    ap.add_argument("--home-state", default=DEFAULT_HOME, help="起位/收尾回的安全位(M100), 默认=训练数据集的 home(各 ep 起始 obs.state 中位数)")
    ap.add_argument("--home-steps", type=int, default=10)
    ap.add_argument("--home-hold-s", type=float, default=1.5)
    ap.add_argument("--save-prefix", default="dual_chunk")
    ap.add_argument("--save-obs", action="store_true", default=False,
                    help="每轮另存 side/eye 单帧 jpg 用于 debug(默认关, 省得堆在项目根)")
    ap.add_argument("--timeout", type=float, default=120.0)
    # 双目 30fps 实时流录制(默认关): 专属抓帧线程读双相机→拼接写视频, 与推理/控制解耦, 平滑不顿
    ap.add_argument("--record-video", action="store_true", default=False,
                    help="保存双相机实时流拼接视频(默认关; 独立抓帧线程, ~30fps 平滑)")
    ap.add_argument("--record-fps", type=float, default=30.0, help="录制视频帧率(默认30, 可与实测 grab fps 对齐)")
    ap.add_argument("--record-path", default="", help="输出视频路径(默认空=存到项目根/outputs/dual_<model>_<toy>_<时间戳>.avi)")
    ap.add_argument("--record-box", action="store_true", default=False,
                    help="录制视频里画服务端返回的红框(默认关=录纯净实时流, 框只在算法侧注入模型输入)")
    ap.add_argument("--stream-fps", type=float, default=30.0, help="相机抓帧目标帧率(仅抓帧线程上限)")
    args = ap.parse_args()

    url = args.url or URLS[args.model]
    task = args.task or task_for(args.model, args.toy)
    display = args.display

    def loop_infer(state, side_bgr, eih_bgr):
        resp = infer_dual(url, side_bgr, eih_bgr, state, task, args.timeout)
        action = np.asarray(resp["action"], dtype=np.float32)
        box = resp.get("box")
        return action, box, resp

    # ---------------- [read-only] 只读核验 ----------------
    if args.read_only and not args.move:
        robot = make_robot_dual(args.port, args.robot_id, args.max_rel, args.cam_side, args.cam_eye)
        robot.connect(calibrate=False)
        rec = make_recorder(args)
        stream = CamStream()
        disp = LiveDisplay(stream, toy=args.toy)
        try:
            state, side_bgr, eih_bgr = read_live_obs_dual(robot, args.save_prefix)
            stream.set(side_bgr, eih_bgr, None)
            if display:
                disp.start()
                time.sleep(0.6)  # 让显示先出一帧原始图
            t0 = time.time()
            action, box, resp = loop_infer(state, side_bgr, eih_bgr)
            roundtrip_ms = (time.time() - t0) * 1000
            print(f"[input ] state={np.round(state, 3).tolist()} task='{task}'")
            print(f"[infer ] {args.model}@{url}  action{action.shape}  box={box}  "
                  f"infer_ms={resp.get('infer_ms')}  roundtrip={roundtrip_ms:.0f}ms")
            cli.print_action_summary(action)
            stream.set_box(box)
            if display:
                print("[display] 'dual_cam' 窗口: 上=side(带红框的模型输入), 下=eye_in_hand(按 q 退出显示)")
                time.sleep(3)
                disp.stop()
            if rec:
                rec.write(stack_dual(side_bgr, eih_bgr, box if args.record_box else None, args.toy))
        finally:
            if rec:
                rec.release()
            robot.disconnect()
        return 0

    # ---------------- [move] 闭环 ----------------
    if not args.move:
        print("! 未给 --move 且未给 --read-only, 默认只干跑(不连臂)。给 --read-only 或 --move。")
        return 2

    robot = make_robot_dual(args.port, args.robot_id, args.max_rel, args.cam_side, args.cam_eye)
    robot.connect(calibrate=False)
    if not robot.is_calibrated:
        print("! 机械臂未校准, 拒绝动臂。")
        robot.disconnect()
        return 2

    exec_n = max(1, min(args.exec_steps, 50))
    home = cli.parse_num(args.home_state)
    stream = CamStream()
    rec = make_recorder(args)
    grab = GrabThread(robot, stream, toy=args.toy, recorder=rec, target_fps=args.stream_fps,
                      record_box=args.record_box)
    disp = LiveDisplay(stream, toy=args.toy)
    # 每轮 action chunk 落盘到 outputs/<run>/ 下(与视频同款命名, 多 run 不覆盖)
    chunk_dir = os.path.join(OUTPUT_DIR, f"dual_{args.model}_{args.toy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(chunk_dir, exist_ok=True)
    print(f"[save] action chunks -> {chunk_dir}")
    grab.start()
    if display:
        disp.start()
    print(f"! 闭环开始: {args.model}@{url} task='{task}'  {args.max_cycles}轮, 每轮执行前{exec_n}步, "
          f"钳制={args.abs_clamp} 发散即停={args.diverge_stop} 平滑={args.smooth} 速率={args.vel_limit}。"
          " 臂连续扫动, torque 保持。请盯窗+备好急停。")

    robot.bus.enable_torque()
    if home is not None:
        try:
            st = cli.drive_to(robot, home, steps=args.home_steps, hold_s=0.2)
            print(f"[起位] 扭矩保持回 home={home}, 起始 state={list(np.round(st, 2))}。")
        except Exception as e:  # noqa: BLE001
            print(f"[起位] 回 home 失败: {type(e).__name__}: {e}")

    aborted = None
    side_bgr = eih_bgr = None
    # ---- 任务完成检测状态(位姿+夹爪轨迹) ----
    grip_hist = []        # 已执行夹爪曲线(采自 stepped[:,5]), 用于检测"张开→闭合=抓过一次"
    grasp_seen = False    # 是否观察到 张开→闭合 的抓取动作
    success_run = 0       # 连续满足成功条件的轮数
    done_cyc = None       # 判定任务完成的 cycle
    # ---- 双目串台守卫状态 ----
    prev_frames = None    # 上一轮 (side, eye), 用于检测抓帧停滞
    stale_run = 0         # 连续"推理输入逐像素未变"的轮数
    try:
        for cyc in range(args.max_cycles):
            stream.wait_first()  # 等抓帧线程首帧到位
            state6 = read_state(robot)  # 总线读关节(等价 get_observation 姿态部分, 不读相机)
            side_bgr, eih_bgr, _ = stream.latest()  # 取最新双帧(≈30fps 流, ≤帧间隔旧)

            # ---- 双目串台守卫: 异常当场落盘 + 中止, 绝不静默把错图喂给模型 ----
            if args.swap_guard and side_bgr is not None and eih_bgr is not None:
                if side_bgr.shape == eih_bgr.shape:
                    fdiff = float(np.mean(np.abs(side_bgr.astype(np.int16) - eih_bgr.astype(np.int16))))
                else:
                    fdiff = -1.0  # 尺寸都不同, 必然不是同一路, 不判串台
                # 抓帧停滞: 连续多轮"推理输入逐像素完全相同"。真实相机有传感器噪声,
                # 两轮完全一致只可能是抓帧线程没在出帧。
                if prev_frames is not None and np.array_equal(side_bgr, prev_frames[0]) \
                        and np.array_equal(eih_bgr, prev_frames[1]):
                    stale_run += 1
                else:
                    stale_run = 0
                prev_frames = (side_bgr.copy(), eih_bgr.copy())

                def _dump_camfail(tag):
                    sp = os.path.join(chunk_dir, f"camfail_{tag}_cyc{cyc}_side.jpg")
                    ep = os.path.join(chunk_dir, f"camfail_{tag}_cyc{cyc}_eye.jpg")
                    cv2.imwrite(sp, side_bgr)
                    cv2.imwrite(ep, eih_bgr)
                    return sp, ep

                if 0 <= fdiff < args.swap_thresh:
                    sp, ep = _dump_camfail("swap")
                    print(f"[cam!] cycle {cyc} 双目串台: 两路画面平均差 {fdiff:.2f} < {args.swap_thresh} —— "
                          f"手眼相机大概率在输出侧视画面。证据已存 {sp} / {ep}。中止。")
                    aborted = cyc
                    break
                if stale_run >= args.stale_warn:
                    sp, ep = _dump_camfail(f"stale{stale_run}")
                    print(f"[cam!] cycle {cyc} 抓帧停滞: 连续 {stale_run + 1} 轮推理输入逐像素完全相同"
                          f" (grab_frames={stream.frames}, 实测应当每轮都涨)。证据已存 {sp} / {ep}。"
                          f"继续跑, 但结果不可信。")
                    stale_run = 0

            if args.save_obs:  # 默认关: 每轮另存 debug 单帧(与录像冗余, 不默认写到项目根)
                cv2.imwrite(f"{args.save_prefix}_observe_{cyc}_side.jpg", side_bgr)
                cv2.imwrite(f"{args.save_prefix}_observe_{cyc}_eye.jpg", eih_bgr)
            if cyc == 0:
                print(f"[cycle {cyc}] 观察: state={np.round(state6, 3).tolist()}  grab_fps≈{grab.measured_fps:.1f}")

            t0 = time.time()
            action, box, resp = loop_infer(state6, side_bgr, eih_bgr)
            wall_ms = (time.time() - t0) * 1000
            np.save(os.path.join(chunk_dir, f"dual_chunk_{cyc}.npy"), action)
            stream.set_box(box)  # 供显示/录制线程画服务端返回的最新框

            stepped = action[:exec_n]
            if args.abs_clamp:
                stepped, clamped, max_dev = cli.clamp_action(stepped)
                body_dev = 0.0
                for j, name in enumerate(cli.MOTOR_ORDER[:-1]):
                    lo, hi = cli.SAFE_ABS_RANGE[name]
                    col = action[:exec_n, j]
                    if col.min() < lo:
                        body_dev = max(body_dev, float(lo - col.min()))
                    if col.max() > hi:
                        body_dev = max(body_dev, float(col.max() - hi))
                if args.diverge_stop and body_dev > args.diverge_margin:
                    print(f"[cycle {cyc}] 执行段身体关节越界 {body_dev:.1f}(>{args.diverge_margin}), 疑似发散, 中止。")
                    aborted = cyc
                    break
                elif clamped:
                    print(f"[cycle {cyc}] 边缘越量 {max_dev:.1f}, 钳制后继续。")

            print(f"[cycle {cyc}] 推理 {wall_ms:.0f}ms(服务器 {resp.get('infer_ms')}ms) box={box} "
                  f"step0={np.round(stepped[0], 3).tolist()}  "
                  f"{exec_n}步区间: lift {stepped[-1,1]:.1f} elf {stepped[-1,2]:.1f} grip {stepped[-1,5]:.1f}")

            if args.vel_limit > 0:
                stepped = cli.velocity_limit(stepped, args.vel_limit,
                                             anchor=np.asarray(state6, dtype=np.float32))

            if args.smooth:
                cli.execute_smooth(robot, stepped, max_step=args.smooth_margin,
                                   step_dt=1 / 30.0, max_sub=args.max_sub)
            else:
                for i in range(min(exec_n, stepped.shape[0])):
                    step = stepped[i]
                    act = {f"{m}.pos": float(v) for m, v in zip(cli.MOTOR_ORDER, step)}
                    robot.send_action(act)
                    time.sleep(1 / 30.0)

            try:
                cur = robot.bus.sync_read("Present_Position")
                st = [float(cur[m]) for m in cli.MOTOR_ORDER]
                print(f"[cycle {cyc}] 执行后 state={np.round(st, 3).tolist()}")
            except Exception as e:  # noqa: BLE001
                print(f"[cycle {cyc}] 读状态失败: {e}")
                continue

            if args.success_detect:
                # 累积夹爪轨迹(用执行段目标曲线, 能捕到 pre-grab 峰值张开)
                grip_hist.extend([float(g) for g in stepped[:, 5]])
                # 检测"张开(>=open)→闭合(<=close)" = 抓过一次
                saw_open = False
                for g in grip_hist:
                    if g >= args.done_grip_open:
                        saw_open = True
                    elif saw_open and g <= args.done_grip_close:
                        grasp_seen = True
                        break
                # 成功 = 抓过 且 臂在"盒子上方释放"位姿 且 夹爪张开
                in_place_pose = (st[1] >= args.done_lift_floor) and (st[2] <= args.done_elbow_ceil)
                if grasp_seen and in_place_pose and st[5] >= args.done_grip_open:
                    success_run += 1
                    if success_run >= args.done_sustain:
                        print(f"[success] cycle {cyc}: 夹爪张开+臂在释放位姿(lift {st[1]:.1f} "
                              f"elbow {st[2]:.1f} grip {st[5]:.1f}), 判定任务完成, 收尾回 home。")
                        done_cyc = cyc
                        break
                else:
                    success_run = 0
    finally:
        if grab.is_alive():
            grab.stop()
            grab.join(timeout=2.0)
        if rec:
            rec.release()
        if display:
            disp.stop()
        if robot.is_connected and home is not None:
            try:
                start = cli.drive_to(robot, home, steps=args.home_steps, hold_s=args.home_hold_s)
                print(f"[收尾] 从 {list(np.round(start,2))} 扭矩保持回 home={home}, 已保持 {args.home_hold_s}s。")
            except Exception as e:  # noqa: BLE001
                print(f"[收尾] 回 home 失败: {type(e).__name__}: {e}")
        try:
            if robot.is_connected:
                robot.disconnect()
        except Exception as e:  # noqa: BLE001
            print(f"[disconnect] {e}")

    return 0


if __name__ == "__main__":
    import sys
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[中断] 已退出(torque 会经 finally 脱开)")
        sys.exit(130)
