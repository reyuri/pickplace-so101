# pickplace-so101

> 基于 SO-ARM101 机械臂的桌面玩具抓取放置 —— 从数据采集、VLA 模型微调、云端推理服务到真机闭环测试的完整链路。

在一个 6 自由度 SO-ARM101 follower 上，用双相机（侧视 + 手眼）采集桌面 3 类玩具的抓取放置示教数据，
分别微调 **SmolVLA-450M** 与 **Octo** 两个 VLA 模型，部署为云端推理服务（RTX 4090）由本地机械臂闭环调用，
并针对真机测试中暴露的问题做了一轮定位与优化。

项目重点不在于「跑通一个 demo」，而在于 **把两个 VLA 模型放在同一套真机环境下做对照**，
并把过程中最耗时的几个问题（跨设备色彩通道不一致、语言条件失效、目标注意力漂移）定位到根因。

---

## 目录

- [硬件与任务](#硬件与任务)
- [数据集](#数据集)
- [系统架构](#系统架构)
- [复现步骤](#复现步骤)
- [SmolVLA 消融 A1 A2 A3](#smolvla-消融-a1-a2-a3)
- [Octo LoRA 微调](#octo-lora-微调)
- [问题定位与优化](#问题定位与优化)
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

**任务**：桌面上摆放一个玩具（大象 / 树 / 球），机械臂将其抓起并放入右侧的木质收纳盒，随后回到初始位姿。
三类玩具各采集一份数据，共用同一套动作基元，差别只在目标物体的外观与位置。

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

数据由 `deploy/record_so101.py` 采集，三份子数据集用 LeRobot 的 `aggregate_datasets()` 合并
（视频走 ffmpeg concat demuxer 流复制，不重编码）。

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

### 一个已知的数据设计缺陷

每个玩具只配了**一条固定指令**（`"Pick up the {toy} and place it into the box"`），
同一玩具的所有 episode 指令完全相同。这意味着 **语言与视觉场景完全冗余** ——
模型只要认出画面里是哪个玩具，就能推断出指令，反之亦然。

这个看起来无害的设计选择，直接导致了后面 Octo 上观察到的「语言条件失效」（见
[问题定位与优化](#问题定位与优化)）。**如果重做一遍，应当给同一物体配多条不同措辞的指令，
并让同一指令跨越不同物体，打断这种一一对应。**

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
│                              SmolVLA-450M (LoRA/全量微调权重)          │
└───────────────────────────────────────────────────────────────────────┘
```

**关键设计：推理服务把「实际喂给模型的那张图」原样回传给本地界面显示。**
A2/A3 在服务端做的红框注入对本地是不可见的，如果不回传，调试时只能靠猜。

---

## 复现步骤

### 0. 采集前：确认相机索引

Windows 的 DSHOW 枚举顺序**会在重插后漂移**，且不报错 —— 只会让画面串台、动作诡异。

```bash
python deploy/dual_camera_viewer.py 1 0      # 参数即 camera index
```

`drive_so101_dual.py` 默认 `--cam-side 1 --cam-eye 0`，并默认开启 `--swap-guard`：
若检测到两路画面疑似对调会直接停机。

```bash
python deploy/record_so101.py                # 遥操作采集 → LeRobot 数据集
```

### 1. 训练

#### SmolVLA

```bash
# 在服务器上（WORK 为工作根目录，默认 /root/autodl-tmp）
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

`smolvla_train_launcher.py` 会**从数据集的 `meta/info.json` 自动探测相机数量**，
生成 `policy.input_features` 并设置 `empty_cameras = 3 - K`
（SmolVLA 的视觉塔固定 3 个槽位，双相机时补 1 个 dummy 槽）：

```
单相机(front)      -> K=1, input={front},        empty=2
双相机(front+hand) -> K=2, input={front,hand},   empty=1   ← 本项目
```

**为什么 `eval_split=0.11`**：135 × 0.11 ≈ 15，即 **120 训练 / 15 验证**。
数据是按玩具分块连续采集的（0-44 大象 / 45-89 树 / 90-134 球），而 lerobot 默认取
「每组的最后 N 条」——那样验证集就全是最后一段的球。所以先跑一次
`train/smolvla/patch_factory_eval.py` 把切分改成按步长均匀抽样，
验证集变成 `{0, 9, 18, ..., 126}`，**恰好跨越三个玩具分段**。

**为什么选 step 5000**：验证 loss 在 5000 步见底、之后回升（过拟合）。
10000 步共约 3.8 个 epoch，5000 步 ≈ **2 epoch**，因此用
`outputs/train_A{1,2,3}/checkpoints/005000/pretrained_model`。

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
python deploy/drive_so101_dual.py --model A2 --toy tree --read-only

# 确认无误后真机执行
python deploy/drive_so101_dual.py --model A2 --toy tree --move --max-cycles 5 --display
```

| 参数 | 含义 |
|---|---|
| `--model` | `A1` / `A2` / `A3` / `OCTO`，对应上表端口 |
| `--toy` | `elephant` / `tree` / `ball`，决定下发的语言指令 |
| `--read-only` / `--move` | 只推理不动机器人 / 真机执行 |
| `--max-cycles` / `--exec-steps` | 闭环轮数 / 每轮执行动作块的前 N 步 |
| `--abs-clamp` | 默认开：关节目标绝对值限幅，防单步大跳 |
| `--diverge-stop` | 检测到动作发散立即停机 |
| `--swap-guard` | 默认开：检测双相机画面疑似对调 |
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

红框由 `data/bake_redbox_dataset.py` 在**训练数据上离线烘焙**（而不是训练时在线注入），
保证训练 / 推理两侧的框在颜色、线宽、标签格式上完全一致：

![A1 与 A2/A3 的输入对比](assets/a2a3_redbox_input.png)

*上排：A1 看到的侧视原图；下排：A2/A3 看到的、烙入目标红框后的图像。*

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

Octo 作为次要对照，用同一批数据微调。

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

**为什么不用全量微调**：135 条轨迹对 204M 参数来说太小，全量微调会过拟合；
LoRA 的低秩增量本身就是很强的正则。

**为什么冻结 attention 的 base kernel 而只训 Q/V**：保留原参数路径
（`Dense_0` / `Dense_1`）可被 Octo 的 `merge_params` 正常加载，
避免「微调后权重合不回原模型」的问题。optimizer 用 optax 的
`multi_transform({"train": adamw, "frozen": set_to_zero})` 保证冻结参数零更新。

本项目采用的是 **step 12000** 的权重。

---

## 问题定位与优化

### 1. ✅ 跨设备色彩通道不一致（RGB / BGR 反了）

**现象**：Octo 微调后验证指标正常，真机上却完全抓不到东西。

**排查**：模型本身没问题，问题出在**推理服务的图像解码路径**——
服务器的 `decode_bgr()` 在读图后又多做了一次通道翻转，导致模型推理时
看到的永远是**红蓝对调**的图，而训练时是正常的 RGB。

**修复**：删掉那次多余的翻转。

**效果**：**同一个 step 12000 的权重，改完立刻就能在真机上正常完成抓取放置。**
训练侧与推理侧的图像预处理必须逐字对齐 —— 这类 bug 不会报错、不会掉点，
只会让真机表现莫名其妙地差。代码里已就地加了注释防止回归：

```python
# NOTE(2026-09-10): 这里原本有 [:, :, ::-1], 是错的, 已删除。
```

### 2. ⚠️ Octo 语言条件失效 → 策略退化成位置先验

**现象**：Octo 能完成抓取放置，但**完全忽略语言指令**——
无论下发的指令是哪个玩具，它**总是去抓离侧视相机最近 / 最显眼的那一个**。

**定位过程**（依次排除）：

| 假设 | 验证 | 结论 |
|---|---|---|
| 指令字符串格式不对 | 直接改 `lang` 字段重跑 | ❌ 排除 |
| T5 tokenizer 16-token 截断 | 缩短指令到截断长度内 | ❌ 排除 |
| 训练数据串了 | 逐条检查 per-toy 指令 | ❌ 排除 |
| **语言与视觉完全冗余** | 量化语言敏感度 | ✅ **根因** |

根因就是[数据集一节](#一个已知的数据设计缺陷)里说的：
**每个玩具一条固定指令**，语言与视觉是一一对应的，模型学「看图」和学「读指令」等价。
而视觉信号更强更好学，于是模型干脆丢掉了语言分支，退化成「抓最显眼的那个」的位置先验。

**量化验证**：构造最小对比对（同图不同指令），测动作输出的差异幅度作为语言敏感度 ——
微调把语言敏感度从 **0.24 压到了 0.156**，即微调**主动削弱**了语言条件。

### 3. ❌ 训练侧修复尝试：对比学习（负结果）

针对上述问题做了一次正面修复尝试：**解冻语言接口 + 在 readout embedding 上加对比损失**
（InfoNCE / hinge margin / 余弦井，`--infonce_w 5.0 --infonce_margin 0.3`），
逼迫同图不同指令产生可区分的表征。

| 指标 | 修复前 | 修复后 |
|---|---|---|
| 语言敏感度 | 0.24（基线 1.82） | **38.38**（涨 21×） |
| 验证 loss | 0.3077 | **0.4418（退化）** |
| 真机表现 | — | **不如 baseline** |

**结论：失败。** 语言敏感度确实被推上去了（21 倍），但验证 loss 明显退化、真机更差。

**为什么失败**（三条，事后复盘）：

1. **对比项扰动的正是动作头的输入**。为了让同图不同指令的输出可区分，损失把扰动直接灌进了
   action head 的输入分布里，等于在教模型「看到不同指令就该输出抖动」，而不是「输出不同的动作」。
2. **只逼「有差异」，没教「差异对应什么」**。InfoNCE 只要求表征可分，不要求分得**对**——
   模型完全可以学到一组任意的、与语义无关的区分方向。
3. **数据层面的冗余没有解决**。在「一物体一指令」的数据上做表征约束，
   等于给一个本来就没有信息量的信号强行注入熵，模型只能去拟合噪声。

**真正的修法应该在数据侧**：让同一物体对应多条不同措辞的指令、让同一指令跨越不同物体
（例如「把最近的放进盒子」「把红色的放进盒子」这类**与视觉解耦**的指令）。
工具链已经就绪（`train/octo/probes/` 下的探针脚本可复用），但没有可用的数据，也就没有继续。

> **这条负结果本身是有价值的**：它说明「用对比损失把语言敏感度刷上去」是一条**看似可行实则有害**的路径 ——
> 敏感度指标涨了 21 倍，真机却更差。指标与目标不一致时，只能信真机。

---

## 真机测试结果

### SmolVLA A1 / A2 / A3 成功率

<!-- TODO(用户填写)：以下的表格骨架已按 A1/A2/A3 × 三类玩具 建好，把实测成功率填进单元格即可 -->

| 模型 | 大象 | 树 | 球 |
|---|---|---|---|
| A1 | 待填 | 待填 | 待填 |
| A2 | 待填 | 待填 | 待填 |
| A3 | 待填 | 待填 | 待填 |

> 单次评测：`--max-cycles 20 --exec-steps 15`，每格 N 次重复，成功判定由 `--success-detect` 自动给出后人工复核。

### 推理耗时

| 环节 | 耗时 |
|---|---|
| SmolVLA 模型前向 | ≈ 300 ms |
| YOLO11n 目标检测（仅 A2/A3） | ≈ 50 ms |
| 网络往返（本地 ↔ 云端） | ≈ 150 ms |
| **单轮合计** | **≈ 500 ms** |

即整条闭环**约 2 Hz**。瓶颈已从模型转到网络往返 —— 若把推理服务下沉到本地，
理论上可降到 ~350 ms。

### 演示视频

<!--
  TODO(用户填写)：把视频文件放进 assets/videos/ 后，取消下面三个 <video> 的注释。
  GitHub 的 README 不直接内嵌播放仓库内的 mp4，建议同时用 GIF 或贴到 issue 里取
  user-attachments 链接；保留 <video> 标签在多数渲染器下也可用。
-->

**1) A1 的问题表现** —— A1 在目标不显著时会去抓错误物体 / 抓空。

<!-- <video src="assets/videos/a1_failure.mp4" controls width="640"></video> -->

**2) A2 / A3 相对 A1 的改进** —— 红框注入后目标锁定稳定，成功率提升。

<!-- <video src="assets/videos/a2a3_success.mp4" controls width="640"></video> -->

**3) Octo 真机测试** —— 能完成抓取放置，但忽略语言指令、固定抓最显眼的那个。

<!-- <video src="assets/videos/octo_language_ignored.mp4" controls width="640"></video> -->

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

| 组件 | 版本 |
|---|---|
| lerobot | 0.4.4（源码 checkout，含本项目对 eval 切分与相机后端的补丁） |
| SmolVLA 基座 | `lerobot/smolvla_base`（450M，SmolVLM2-500M 骨干） |
| Octo | 官方 checkpoint + 自定义 LoRA fork（Q/V 适配器 + 可加载的 base kernel 路径） |
| YOLO | ultralytics YOLO11n |
| 推理硬件 | AutoDL RTX 4090（同时常驻 3 个 SmolVLA + 1 个 Octo + 1 个 YOLO） |

数据转换侧最低依赖：`pyarrow`、`av`、`numpy`（**不需要 torchcodec**）。

### 两处失败模式备忘

- **相机索引会漂移**：Windows DSHOW 的枚举顺序重插后会变，且不报错。
  每次开机先跑 `dual_camera_viewer.py` 目视确认；`drive_so101_dual.py` 默认开 `--swap-guard`。
- **训练 / 推理预处理必须逐字对齐**：`decode_bgr` 的多余翻转曾让模型在红蓝对调下跑了整个训练，
  指标看不出来、真机全废。两侧的图像处理代码请当成同一份代码来维护。

---

## License

Apache-2.0
