# Xmimic 动作跟踪代码

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)


## 安装

- 根据 [Isaac Lab 安装指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)安装 Isaac Lab，并将环境版本对齐到以下当前验证版本。推荐使用 conda 方式安装，便于后续在终端中直接调用 Python 脚本。

当前验证环境:

- Python 3.11
- Isaac Sim 5.1

- 将本仓库单独克隆到 Isaac Lab 目录外，例如不要放在 `IsaacLab` 文件夹内部:

```bash
# 方式一: SSH
git clone git@github.com:Open-X-Humanoid/xMimic.git

# 方式二: HTTPS
git clone https://github.com/Open-X-Humanoid/xMimic.git
```

- 安装项目依赖:

```bash
python -m pip install -e source/whole_body_tracking
```

## 动作跟踪

### 动作预处理

将处理后的动作数据保存在本地，例如 `motion_data/` 目录下，并在训练脚本中指定对应的动作文件。参考动作需要完成重映射，并且仅使用广义坐标。

- 准备参考动作数据集，请遵守原始数据集的许可证。这里使用与 Unitree 数据集中 `.csv` 文件一致的格式约定。

    - Unitree 重映射后的 LAFAN1 数据集可从 [HuggingFace](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) 下载。
    - Sidekick 动作来自 [KungfuBot](https://kungfu-bot.github.io/)。
    - Christiano Ronaldo 庆祝动作来自 [ASAP](https://github.com/LeCAR-Lab/ASAP)。
    - 平衡动作来自 [HuB](https://hub-robot.github.io/)。

- 通过正向运动学将重映射后的动作转换为包含最大坐标信息的数据，包括 body pose、body velocity 和 body acceleration。脚本会在本地保存 `{motion_name}.npz` 文件，默认输出到 `motion_data/`:

```bash
python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 \
--output_name {motion_name} --output_dir motion_data --headless
```

- 在 Isaac Sim 中回放转换后的动作，用于检查动作是否正常:

```bash
python scripts/replay_npz.py --motion_file motion_data/{motion_name}.npz
```
## 训练仿真全流程
### 动作重映射

下载 xGMR，使用 xGMR 将 BVH 格式数据转化为 PKL 格式数据。

```bash
git clone https://github.com/Open-X-Humanoid/xGMR.git
```
动作重映射
```bash
python scripts/bvh_to_robot.py --bvh_file data/lafan/dance1_subject3.bvh --robot dex_evt2 --save_path output/YYW.pkl --motion_fps {fps} --add_collision_avoidance True --format "lafan"
```

将生成的 `.pkl` 文件复制到 `motion_data/` 文件夹下。

### 动作格式转换

- 将 xGMR 输出的 `.pkl` 文件转化成训练所需的 `.npz` 文件:

```bash
python scripts/gmr_to_npz_inter.py \
--input_file " " \
--input_fps {fps} --frame_range 10 -1 --output_name {motion_name} --output_fps 100 \
--robot dex_evt \
--start_frames 100 --end_frames 50 --hold_pos 300
```

### 策略训练

- 指定运动文件训练 policy:

```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-DexEVT-Wo-State-Estimation-v0 \
--num_envs 4096 \
--max_iterations 100000 \
--device cuda:0 \
--motion_file motion_data/{motion_name}.npz \
--headless --logger tensorboard 
```

### 策略评估

- 使用以下命令运行训练好的 policy:

```bash
python scripts/rsl_rl/play.py \
--task=Tracking-Flat-DexEVT-Wo-State-Estimation-v0 --num_envs=2 \
--load_run={run_folder_regex} --checkpoint={checkpoint_regex} \
--motion_file motion_data/{motion_name}.npz
```

### 启动 MuJoCo 仿真

下载 xSIM_MUJOCO:

```bash
git clone https://github.com/Open-X-Humanoid/xSIM_MUJOCO.git
```

启动 MuJoCo 仿真:

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=your_domain_id
python scripts/simulator_view_asyn.py -m evt2(机器人名称)
```

### 策略部署

部署代码 xMIGCS 地址:

```bash
git clone https://github.com/Open-X-Humanoid/xMIGCS.git
```

将导出的 `.onnx` 文件放在 `policy/beyond_mimic/model` 中，同时在 `policy/beyond_mimic/config/BeyondMimic_down.yaml` 中配置对应的 `onnx_path`、`kp` 和 `kd`。

在 xMIGCS 目录下执行:

```bash
export ROS_DOMAIN_ID=your_domain_id
python3 rl_control_node.py
```

若使用键盘控制，按 `1`；若使用 Xbox 手柄控制，按 `LT+HOME` 进入 Xmimic 对应策略。

## 代码结构

下面是本仓库的代码结构概览:

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`**
  该目录包含定义 BeyondMimic MDP 的基础函数，主要模块如下:

    - **`commands.py`**
      命令库，用于根据参考动作、当前机器人状态和误差计算相关变量，包括位姿误差、速度误差、初始状态随机化和自适应采样。

    - **`rewards.py`**
      实现 DeepMimic 奖励函数和相关平滑项。

    - **`events.py`**
      实现域随机化相关项。

    - **`observations.py`**
      实现动作跟踪和数据采集所需的观测项。

    - **`terminations.py`**
      实现提前终止和超时条件。

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`**
  包含动作跟踪任务的环境 MDP 超参数配置。

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`**
  包含动作跟踪任务的 PPO 超参数配置。

- **`source/whole_body_tracking/whole_body_tracking/robots`**
  包含机器人相关配置，包括 armature 参数、关节刚度和阻尼计算，以及动作缩放计算。

- **`scripts`**
  包含动作数据预处理、策略训练和策略评估相关的工具脚本。

该结构便于保持模块化，也方便后续开发者扩展和定位代码。
