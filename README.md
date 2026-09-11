# pickplace-so101

> 基于 SO-ARM101 机械臂的桌面玩具抓取放置 —— 从数据采集、VLA 模型微调、云端推理服务到真机闭环测试的完整链路。

在一个 6 自由度 SO-ARM101 follower 上，用双相机（侧视 + 手眼）采集桌面 3 类玩具的抓取放置示教数据，
分别微调 **SmolVLA-450M** 与 **Octo** 两个 VLA 模型，部署为云端推理服务（RTX 4090）由本地机械臂闭环调用，
并针对真机测试中暴露的问题做定位与优化。

---

## 目录

- [硬件与任务](#硬件与任务)
- [数据集](#数据集)
- [系统架构](#系统架构)
- [复现步骤](#复现步骤)
- [SmolVLA 消融 A1 A2 A3](#smolvla-消融-a1-a2-a3)
- [Octo LoRA 微调](#octo-lora-微调)
- [真机测试结果](#真机测试结果)
- [仓库结构](#仓库结构)
- [环境](#环境)

---

## 硬件与任务

| 项 | 配置 |
|---|---|
| 机械臂 | SO-ARM101 follower，6 自由度（shoulder_pan / shoulder_lift / elbow_flex / wrist_flex / wrist_roll / gripper） |
| 相机 1 | 侧视 USB 相机（Logitech Brio 90），640×480 @ 30 fps，MJPG |
| 相机 2 | 手眼相机（装在腕部），640×480 @ 30 fps，MJPG |
| 上位机 | Windows，机械臂串口 `COM3` |
| 训练 / 推理 | 云端 AutoDL RTX 4090，SSH 隧道回传 |

**任务**：如下视频所示，桌面上同时摆放三个玩具（大象 / 树 / 球）和收纳盒，机械臂按照指令抓起目标玩具并放入收纳盒。

![机械臂抓取放置任务演示](assets/videos/task_overview.gif)

*一个完整的抓取放置周期。画面为双相机拼接：上半 **SIDE**（侧视）、下半 **EYE_IN_HAND**（手眼）。*

---

## 数据集

**[reyqiao/lerobot_reyqiao_toy](https://huggingface.co/datasets/reyqiao/lerobot_reyqiao_toy)** — HuggingFace Datasets

| 指标 | 数值 |
|---|---|
| Episode 数 | 135（大象 / 树 / 球 各 45） |
| 总帧数 | 35,556 |
| 总时长 | ≈ 19.8 min @ 30 fps |
| 相机 | 2 路（`observation.images.side` + `observation.images.eye_in_hand`），640×480 |
| 状态 / 动作 | 6 维关节绝对位置（float32） |
| 机器人类型 | `so_follower` |
| 格式 | LeRobot **v3.0** |
| 体积 | ≈ 413 MB |

```python
from lerobot.datasets.aggregate import aggregate_datasets

aggregate_datasets(
    repo_ids=["local/lerobot_rey_toy_ball", "local/lerobot_rey_toy_elephant", "local/lerobot_rey_toy_tree"],
    aggr_repo_id="reyqiao/lerobot_reyqiao_toy",
    roots=[ROOT / s for s in SRC],
    aggr_root=OUT,
)
```

> 注意用 `aggregate_datasets()` 而不是 `merge_datasets()`：后者在末尾会把结果重新以
> `LeRobotDataset` 打开一遍，那条路径依赖 `torchcodec`；前者只用 `pyav` + `ffmpeg`。

---

## 系统架构

```
┌─────────────────────────── 本地 (Windows) ───────────────────────────┐
│                                                                      │
│  双相机 ──┬─→ dual_camera_viewer.py   目视确认相机索引（每次开机必做）│
│           │                                                          │
│           ├─→ record_so101.py         遥操作采集 → LeRobot 数据集     │
│           │                                                          │
│           └─→ drive_so101_dual.py     ★ 真机闭环驱动                  │
│                    │  ① 读 6 维关节状态 + 双相机帧                     │
│                    │  ② POST /infer ──────────────────┐              │
│                    │  ③ 收到动作块，执行前 N 步         │              │
│                    │  ④ 显示「真正进入模型的那张 side 图」│              │
│                    └───────────────────────────────────┼──────────────│
└────────────────────────────────────────────────────────┼─────────────┘
                       SSH 隧道  -L 6010/6011/6012        │
┌──────────────────────── 云端 (AutoDL RTX 4090) ─────────┼─────────────┐
│                                                          ▼            │
│   drive_so101_dual.py ──→ serve_smolvla_dual.py  :6010 A1 / :6011 A2  │
│                                                  :6012 A3 / :6020 Octo │
│                                        │                              │
│                                        ├─(A2/A3)→ yolo_serve.py :6090 │
│                                        │          YOLO11n 检测目标     │
│                                        │          并把红框烙进 side 帧  │
│                                        ▼                              │
│                              SmolVLA-450M (微调权重)               │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 复现步骤

### 0. 采集

```bash
python deploy/record_so101.py                # 遥操作采集 → LeRobot 数据集
```

### 1. 训练

#### SmolVLA

```bash
# 在服务器上
bash train/smolvla/train_A1.sh               # 基线，侧视原图
bash train/smolvla/train_A2.sh               # 侧视注入 YOLO 红框
bash train/smolvla/train_A3.sh               # 红框 + 框感知指令

# 后台跑
nohup bash train/smolvla/train_A1.sh > train_A1.log 2>&1 &
```

三个脚本**超参完全一致**，只有数据集与输出目录不同：

```bash
python smolvla_train_launcher.py \
  --dataset.root=$WORK/lerobot_rey_toy_all \
  --steps=10000 --batch_size=12 --num_workers=4 \
  --eval_steps=1000 --save_freq=1000 \
  --dataset.eval_split=0.11 \
  --output_dir=$WORK/outputs/train_A1 --job_name=toy_all_A1
```

#### Octo

```bash
bash train/octo/run_finetune.sh
```

### 2. 部署

```bash
# 服务器：起 YOLO + A1/A2/A3 四个服务
bash deploy/serve_dual.sh start
bash deploy/serve_dual.sh status

# 本地：建立 SSH 隧道（Windows 上用 netstat -ano 判活，pgrep 匹配不到 ssh.exe）
bash deploy/tunnel_dual.sh start
```

### 3. 真机测试

```bash
# 先干跑（不动机器人），只打印动作与耗时
python deploy/drive_so101_dual.py --model A1 --toy tree --read-only

# 确认无误后真机执行
python deploy/drive_so101_dual.py --model A1 --toy tree --move --max-cycles 20 --display
```

| 参数 | 含义 |
|---|---|
| `--model` | `A1` / `A2` / `A3` / `OCTO`，对应上表端口 |
| `--toy` | `elephant` / `tree` / `ball`，决定下发的语言指令 |
| `--read-only` / `--move` | 只推理不动机器人 / 真机执行 |
| `--max-cycles` / `--exec-steps` | 闭环轮数 / 每轮执行动作块的前 N 步 |
| `--abs-clamp` | 默认开：关节目标绝对值限幅，防单步大跳 |
| `--diverge-stop` | 检测到动作发散立即停机 |
| `--success-detect` | 依据夹爪开合与抬升高度自动判成功 |

> ⚠️ 真机运动必须有人在场、急停就绪。`--read-only` 先跑通再 `--move`。

---

## SmolVLA 消融 A1 A2 A3

三个变体的**训练超参完全相同**，只改输入表征：

| 变体 | 数据集 | 侧视图输入 | 语言指令 |
|---|---|---|---|
| **A1** | `lerobot_rey_toy_all` | 原图 | `Pick up the {toy} and place it into the box` |
| **A2** | `..._redbox` | **YOLO11n 检出目标后烙入红框** | 同上（不变） |
| **A3** | `..._redbox_a3` | 同 A2 | `Pick up the {toy} **in red box** and place it into the box` |

动机：双相机下侧视视角目标很小，模型容易把注意力放到机械臂自身运动上。
用 YOLO 把目标直接标注在输入上，等于**把「看哪里」这件事从模型里摘出去**，
让策略网络专注学「怎么动」。A3 进一步用语言显式指代红框，测试语言是否能与视觉线索绑定。

![A1 与 A2/A3 的侧视输入对比](assets/a2a3_redbox_input.png)

*同一帧画面：左为 A1 看到的侧视原图，右为 A2/A3 看到的、烙入 YOLO11n 目标框后的图。*

### 训练结果

三个变体在 step 5000（验证 loss 最低点）的验证 loss：

| 变体 | 验证 loss @ 5000 |
|---|---|
| A1 | 0.2832 |
| A2 | 0.2907 |
| A3 | 0.2923 |

**三者统计上无显著差异，A1 反而最低。** 这说明验证 loss 这个指标**无法区分**
「模型是否真的在看目标」—— 它与真机成功率不是一回事，消融结论必须由真机 rollout 给出。

---

## Octo LoRA 微调

Octo 用同一批数据微调。

### 数据格式转换

Octo 用的是自己的一套数据管线，需要先把 LeRobot 数据集转成逐 episode 的 npz：

```bash
# LeRobot v3.0  →  octo_frames/<toy>/ep_<N>.npz
python data/lerobot_to_octo_frames.py \
    --root /path/to/lerobot_rey_toy_elephant --toy elephant \
    --out  /path/to/octo_frames --size 256

# 自查：逐条比对 npz 的 T 与 meta 里的 episode length
python data/lerobot_to_octo_frames.py --root ... --toy ... --out ... --verify-only
```

每个 npz 含 `side` / `wrist` / `action` / `state` / `lang`。
转换器**刻意不依赖 `torchcodec`** —— 直接 `pyarrow` 读 parquet + `PyAV` 解码 mp4，
只需 `pyarrow` 和 `av` 两个纯 pip 依赖，在 Windows 和服务器上都能跑。

再由 `data/lerobot_toy_dataset.py`（TFDS builder `LerobotToy`）与 `data/build_tfds.py` 打包成 TFDS。

### 训练配置

`train/octo/finetune_octo_lora_qv.py`：

| 项 | 值 |
|---|---|
| 微调方式 | LoRA rank 8，**仅挂在 attention 的 query / value 上** |
| 解冻部分 | DiffusionActionHead 全量训练（保留 OpenX 预训练的 dim7/horizon4 头） |
| 冻结部分 | ViT、T5 语言塔、attention base kernel、LayerNorm |
| window / horizon | 2 / 4 |
| 图像尺寸 | primary 256×256，wrist 128×128 |
| batch / lr / steps | 8 / 1e-4 / 16000 |

---

## 真机测试结果

### SmolVLA A1 / A2 / A3 成功率

每个模型各跑 **15 次真机测试**（大象 / 树 / 球 每类 5 次），单次 `--max-cycles 20 --exec-steps 15`。
下表是这 15 次的逐次明细,M,L,R分别摆放在中间，左边，右边。

| 目标玩具 | 第 N 次 | A1 | A2 | A3 |
|:--|:-:|:-:|:-:|:-:|
| Tree(M) | 1 | ❌ | ✅ | ✅ |
| Tree(M) | 2 | ✅ | ❌ | ✅ |
| Tree(M) | 3 | ✅ | ✅ | ✅ |
| Tree(L) | 4 | ✅ | ✅ | ✅ |
| Tree(R) | 5 | ✅ | ✅ | ✅ |
| **Tree 小计** | **5** | **4 / 5** | **4 / 5** | **5 / 5** |
| Ball(M) | 1 | ❌ | ✅ | ✅ |
| Ball(M) | 2 | ❌ | ❌ | ❌ |
| Ball(M) | 3 | ❌ | ✅ | ✅ |
| Ball(L) | 4 | ❌ | ✅ | ✅ |
| Ball(R) | 5 | ❌ | ❌ | ❌ |
| **Ball 小计** | **5** | **0 / 5** | **3 / 5** | **3 / 5** |
| Elephant(M) | 1 | ❌ | ✅ | ✅ |
| Elephant(M) | 2 | ✅ | ✅ | ✅ |
| Elephant(M) | 3 | ❌ | ❌ | ✅ |
| Elephant(L) | 4 | ❌ | ✅ | ✅ |
| Elephant(R) | 5 | ❌ | ✅ | ❌ |
| **Elephant 小计** | **5** | **1 / 5** | **4 / 5** | **4 / 5** |

**15 次测试的合计成功率（不分玩具）：**

| 模型 | 成功次数 | 测试次数 | 成功率 |
|:--|:-:|:-:|:-:|
| **A1** | **5** | 15 | **33.3 %** |
| **A2** | **11** | 15 | **73.3 %** |
| **A3** | **12** | 15 | **80.0 %** |

> A1 → A2 提升 **+40.0** 个百分点，来自把「目标在哪」从模型里摘出去；
> A2 → A3 只再涨 **+6.7** 个百分点，说明额外的语言指代（"in red box"）带来轻微收益 ——
> 但主要增益来自视觉注意力的显式注入。
> 分玩具看：A1 在 ball 上 **0 / 5**（目标最小、最不好抓取），A2/A3 把它拉到 3 / 5 以上。

### 演示视频

以下均为真机实拍，画面为双相机拼接：上半 **SIDE**（侧视）、下半 **EYE_IN_HAND**（手眼）。

#### 1) A1 的动作捷径问题

指令是 `Pick up the ball and place it into the box`（抓左侧的球），
A1 却抓成了正前方的 tree。**A1 过拟合了示教轨迹**，
无法有效响应 `pick up [具体物体]` 的文本指令，导致抓错物体或空抓。

![A1 动作捷径：指令要求抓 ball，实际抓成了 tree](assets/videos/a1_ball_shortcut.gif)

#### 2) A2：用 YOLO11n 给 SmolVLA 注入「视觉注意力」

同一条指令、同一套初始摆放，A2 先用 YOLO11n 检出目标、把红框烙进侧视图再送入策略网络，
成功抓到了左侧的 ball。

![A2 红框注入后正确抓取 ball](assets/videos/a2_ball_redbox.gif)

#### 3) Octo：忽略语言指令，按视觉显著性选目标

指令是抓 elephant，Octo 却去抓了左侧的 ball —— 虽然抓取与放置动作本身都成功完成了。
说明其语言条件已退化，策略实际上在按「哪个目标更显眼」选择物体。

![Octo 忽略语言指令，改抓了 ball](assets/videos/octo_elephant_ignores_lang.gif)

### 推理耗时

云端推理服务器为 RTX 4090 24G。两个模型都跑「云端一次规划 → 本地执行若干步」的闭环。

**SmolVLA（A1 / A2 / A3）**

| 环节 | 耗时 |
|---|---|
| 模型一次前向 | ≈ 300 ms |
| YOLO11n 目标检测（仅 A2/A3） | ≈ 40 ms |
| 网络往返（本地 ↔ 云端） | ≈ 140 ms |
| **单轮规划** | **≈ 480 ms** |

每轮规划输出一个动作块，本地执行其中的前 **15 步**；完成一次完整的抓取放置约需 **13 ~ 15 轮**规划，
即整体耗时约 **6 ~ 7 s**。

**Octo**

| 环节 | 耗时 |
|---|---|
| 模型一次前向 | ≈ 130 ms |
| 网络往返（本地 ↔ 云端） | ≈ 50 ms |
| **单轮规划** | **≈ 180 ms** |

Octo 单轮只走 **4 步**，完成同一任务需要约 **55 ~ 60 轮**规划，整体约 **10~11 s**。
它的单轮比 SmolVLA 快约 2.7 倍，但每轮推进的步数只有前者的约 1/4 ——
**每轮规划更便宜，代价是轮数多得多，任务整体反而更慢。**

---

## 仓库结构

```
pickplace-so101/
├── data/                          数据：格式转换与数据集构建
│   ├── lerobot_to_octo_frames.py     LeRobot → Octo npz（不依赖 torchcodec）
│   ├── lerobot_toy_dataset.py        Octo TFDS builder
│   ├── build_tfds.py                 打包 TFDS
│   ├── bake_redbox_dataset.py        离线烘焙 YOLO 红框进训练数据
│   └── make_a3_dataset.py            生成 A3 的框感知指令数据集
│
├── train/
│   ├── smolvla/                   SmolVLA 微调
│   │   ├── smolvla_train_launcher.py  自动探测相机数，生成 input_features
│   │   ├── train_A1.sh / A2.sh / A3.sh
│   │   └── tb_sidecar.py              TensorBoard 日志旁路
│   └── octo/
│       ├── finetune_octo_lora_qv.py   LoRA(Q/V) 微调
│       ├── run_finetune.sh / run_infonce.sh
│       └── probes/                    语言敏感度探针与对比实验
│
├── deploy/                        推理服务与真机驱动
│   ├── serve_smolvla_dual.py         双相机 SmolVLA 推理服务
│   ├── serve_octo_native.py          Octo 推理服务
│   ├── serve_dual.sh                 一键起 YOLO + A1/A2/A3
│   ├── tunnel_dual.sh                本地 SSH 隧道管理
│   ├── drive_so101_dual.py           ★ 真机闭环驱动（含多项安全护栏）
│   ├── record_so101.py               遥操作数据采集
│   └── dual_camera_viewer.py         开机前相机索引确认
│
├── yolo/                          YOLO11n 目标检测
│   ├── yolo_train.py                 训练（5 类：elephant/tree/ball/box/arm）
│   ├── yolo_serve.py                 推理服务，供 A2/A3 调用
│   ├── gen_yolo_targetbox_sidecar.py 生成目标框 sidecar
│   ├── prepare_yolo_ann.py           抽帧供标注
│   └── resample_annot_diverse.py     最远点采样选标注帧（200 train / 20 val）
│
├── assets/                       图片与演示视频
└── README.md
```

---

## 环境

**本地与服务器的 lerobot 不是同一个版本**，两个补丁也打在不同的 checkout 上：

| 侧 | lerobot | 用途 | 打的补丁 |
|---|---|---|---|
| 本地（Windows） | **0.4.4**（源码 checkout，editable 安装） | 数据采集、真机闭环驱动 | `cameras/utils.py`：`CAP_MSMF` → `CAP_DSHOW`（Windows 相机后端） |
| 云端（AutoDL） | **0.6.2**（conda env `smolvla`） | 训练、推理服务 | `datasets/factory.py`：eval 切分改为均匀抽样（`train/smolvla/patch_factory_eval.py`） |

| 其他组件 | 说明 |
|---|---|
| SmolVLA 基座 | `lerobot/smolvla_base`（450M，SmolVLM2-500M 骨干）；A1/A2/A3 为**微调**，非 LoRA |
| Octo | 官方 checkpoint + 自定义 LoRA fork（Q/V 适配器 + 可加载的 base kernel 路径） |
| YOLO | ultralytics YOLO11n |
| 推理硬件 | AutoDL RTX 4090（同时常驻 3 个 SmolVLA + 1 个 Octo + 1 个 YOLO） |

数据转换侧最低依赖：`pyarrow`、`av`、`numpy`（**不需要 torchcodec**）。

---

## License

Apache-2.0
