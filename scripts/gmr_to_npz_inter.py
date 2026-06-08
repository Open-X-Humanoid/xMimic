"""This script replay a motion from a npz/pkl file and output it to a npz file

.. code-block:: bash

    # Usage
    python gmr_to_npz.py --input_file /path/to/motion.npz --input_fps 30 --frame_range 122 722 \
    --output_file ./motions/dance1_subject2.npz --output_fps 50
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import types

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay motion from npz/pkl file and output to npz file.")
parser.add_argument("--input_file", type=str, required=True, help="The path to the input motion npz/pkl file.")
parser.add_argument("--input_fps", type=int, default=30, help="The fps of the input motion.")
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help=(
        "frame range: START END (both inclusive). The frame index starts from 1. If not provided, all frames will be"
        " loaded."
    ),
)
parser.add_argument("--output_name", type=str, required=True, help="The name of the motion npz file.")
parser.add_argument(
    "--output_dir",
    type=str,
    default="motion_data",
    help="Directory where the converted motion npz file will be stored.",
)
parser.add_argument("--output_fps", type=int, default=50, help="The fps of the output motion.")
parser.add_argument(
    "--robot",
    type=str,
    default="dex_evt",
    choices=["dex_evt"],
    help="Robot type: dex_evt (default: dex_evt).",
)
parser.add_argument("--knee_modify", action="store_true", help="Apply special knee interpolation for start/end frames.")
parser.add_argument("--start_frames", type=int, default=0, help="Number of frames to interpolate at start.")
parser.add_argument("--end_frames", type=int, default=0, help="Number of frames to interpolate at end.")
parser.add_argument(
    "--correct_root_pose_coupled",
    action="store_true",
    help="Apply coupled root pose correction (position + yaw) before interpolation.",
)
parser.add_argument(
    "--hold_pos",
    type=int,
    default=0,
    help="If > 0, append this many frames holding the default pose after interpolation.",
)
parser.add_argument(
    "--split_frame",
    type=int,
    default=None,
    help="If set, split the processed motion at this frame index (before/after) and save two NPZ files.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

##
# Pre-defined configs
##
from whole_body_tracking.robots.dex_evt import DEX_EVT_CFG



def _ensure_numpy_core_compatibility():
    """Register numpy._core aliases for legacy pickle compatibility."""
    import sys

    if "numpy._core" in sys.modules:
        return

    core_module = getattr(np, "core", None)
    if core_module is None:
        return

    shim = types.ModuleType("numpy._core")
    shim.__dict__.update(core_module.__dict__)
    sys.modules["numpy._core"] = shim
    for name in ("multiarray", "umath", "numerictypes", "_multiarray_umath"):
        submodule = getattr(core_module, name, None)
        if submodule is not None:
            sys.modules[f"numpy._core.{name}"] = submodule


def correct_root_pose_coupled(root_pos: np.ndarray, root_rot_xyzw: np.ndarray, target_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply coupled translation + yaw correction on the root pose."""
    print("\n=== 耦合修正根节点位姿 ===")
    root_pos = np.asarray(root_pos, dtype=np.float64)
    root_rot_xyzw = np.asarray(root_rot_xyzw, dtype=np.float64)
    target_pos = np.asarray(target_pos, dtype=np.float64)

    initial_pos = root_pos[0].copy()
    initial_quat_xyzw = root_rot_xyzw[0].copy()
    initial_rot = R.from_quat(initial_quat_xyzw)
    print(f"原始初始位置: {initial_pos}")
    print(f"原始初始四元数 [x,y,z,w]: {initial_quat_xyzw}")

    target_rot = R.identity()
    initial_euler = initial_rot.as_euler("zyx", degrees=False)
    target_euler = target_rot.as_euler("zyx", degrees=False)
    yaw_correction = target_euler[0] - initial_euler[0]

    correction_rot = R.from_euler("z", yaw_correction)
    correction_T = np.eye(4, dtype=np.float64)
    correction_T[:3, :3] = correction_rot.as_matrix()
    correction_T[:3, 3] = target_pos - correction_rot.apply(initial_pos)

    print(f"yaw方向修正角度: {np.degrees(yaw_correction):.3f} 度")
    print("修正旋转矩阵:")
    print(correction_T[:3, :3])
    print(f"修正平移向量: {correction_T[:3, 3]}")

    corrected_positions = []
    corrected_rotations = []
    for i in range(len(root_pos)):
        current_T = np.eye(4, dtype=np.float64)
        current_T[:3, :3] = R.from_quat(root_rot_xyzw[i]).as_matrix()
        current_T[:3, 3] = root_pos[i]
        corrected_T = correction_T @ current_T
        corrected_positions.append(corrected_T[:3, 3])
        corrected_rotations.append(R.from_matrix(corrected_T[:3, :3]).as_quat())

    corrected_positions = np.asarray(corrected_positions, dtype=np.float32)
    corrected_rotations = np.asarray(corrected_rotations, dtype=np.float32)
    print(f"修正后初始位置: {corrected_positions[0]}")
    print(f"修正后初始四元数 [x,y,z,w]: {corrected_rotations[0]}")
    return corrected_positions, corrected_rotations

##
# Robot Configuration System
##
# 仅保留 Dex EVT 配置。

DEX_EVT_INPUT_JOINT_ORDER = [
    "hip_pitch_l_joint", "hip_roll_l_joint", "hip_yaw_l_joint",
    "knee_pitch_l_joint", "ankle_pitch_l_joint", "ankle_roll_l_joint",
    "hip_pitch_r_joint", "hip_roll_r_joint", "hip_yaw_r_joint",
    "knee_pitch_r_joint", "ankle_pitch_r_joint", "ankle_roll_r_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "shoulder_pitch_l_joint", "shoulder_roll_l_joint", "shoulder_yaw_l_joint",
    "elbow_pitch_l_joint", "elbow_yaw_l_joint", "wrist_pitch_l_joint", "wrist_roll_l_joint",
    "shoulder_pitch_r_joint", "shoulder_roll_r_joint", "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint", "elbow_yaw_r_joint", "wrist_pitch_r_joint", "wrist_roll_r_joint",
]

DEX_EVT_INPUT_JOINT_ORDER_WITH_HEAD = [
    "hip_pitch_l_joint", "hip_roll_l_joint", "hip_yaw_l_joint",
    "knee_pitch_l_joint", "ankle_pitch_l_joint", "ankle_roll_l_joint",
    "hip_pitch_r_joint", "hip_roll_r_joint", "hip_yaw_r_joint",
    "knee_pitch_r_joint", "ankle_pitch_r_joint", "ankle_roll_r_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "head_yaw_joint", "head_pitch_joint",
    "shoulder_pitch_l_joint", "shoulder_roll_l_joint", "shoulder_yaw_l_joint",
    "elbow_pitch_l_joint", "elbow_yaw_l_joint", "wrist_pitch_l_joint", "wrist_roll_l_joint",
    "shoulder_pitch_r_joint", "shoulder_roll_r_joint", "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint", "elbow_yaw_r_joint", "wrist_pitch_r_joint", "wrist_roll_r_joint",
]

# Dex EVT机器人配置 (23 DOF)
ROBOT_CONFIG_DEX_EVT = {
    "default_pose": [
        0.0, 0.0, 0.95,  # root position
        1.0, 0.0, 0.0, 0.0,  # root rotation (wxyz)
        # left leg
        -0.25, 0.0, 0.0, 0.5, -0.25, 0.0,
        # right leg
        -0.25, 0.0, 0.0, 0.5, -0.25, 0.0,
        # waist
        0.0, 0.0, 0.0,
        # left arm
        0.0, 0.3, 0.0, -0.3,
        # right arm
        0.0, -0.3, 0.0, -0.3,
    ],
    "joint_names": [
        "hip_pitch_l_joint",
        "hip_roll_l_joint",
        "hip_yaw_l_joint",
        "knee_pitch_l_joint",
        "ankle_pitch_l_joint",
        "ankle_roll_l_joint",
        "hip_pitch_r_joint",
        "hip_roll_r_joint",
        "hip_yaw_r_joint",
        "knee_pitch_r_joint",
        "ankle_pitch_r_joint",
        "ankle_roll_r_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "shoulder_pitch_l_joint",
        "shoulder_roll_l_joint",
        "shoulder_yaw_l_joint",
        "elbow_pitch_l_joint",
        "shoulder_pitch_r_joint",
        "shoulder_roll_r_joint",
        "shoulder_yaw_r_joint",
        "elbow_pitch_r_joint",
    ],
    "isaac_lab_cfg": DEX_EVT_CFG,
    "num_dof": 23,
    "height_offset": 0.0,
    "root_target_pos": [0.0, 0.0, 0.95],
    "input_joint_order": DEX_EVT_INPUT_JOINT_ORDER,
    "input_joint_orders": {
        29: DEX_EVT_INPUT_JOINT_ORDER,
        31: DEX_EVT_INPUT_JOINT_ORDER_WITH_HEAD,
    },
}

ROBOT_CONFIGS = {
    "dex_evt": ROBOT_CONFIG_DEX_EVT,
}

def get_robot_config(robot_type: str):
    """获取机器人配置
    
    Args:
        robot_type: 机器人类型,仅支持 "dex_evt"
        
    Returns:
        包含机器人所有配置的字典
        
    Raises:
        ValueError: 如果机器人类型不支持
    """
    if robot_type not in ROBOT_CONFIGS:
        raise ValueError(
            f"Unknown robot type: '{robot_type}'. "
            f"Supported types: {list(ROBOT_CONFIGS.keys())}"
        )
    return ROBOT_CONFIGS[robot_type]

def create_scene_cfg(robot_type: str = "dex_evt") -> InteractiveSceneCfg:
    """根据机器人类型创建场景配置
    
    Args:
        robot_type: 机器人类型,仅支持 "dex_evt"
        
    Returns:
        场景配置对象
    """
    robot_config = get_robot_config(robot_type)
    
    @configclass
    class ReplayMotionsSceneCfg(InteractiveSceneCfg):
        """Configuration for a replay motions scene."""

        # ground plane
        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

        # lights
        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                intensity=750.0,
                texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            ),
        )

        # articulation - Dex EVT
        robot: ArticulationCfg = robot_config["isaac_lab_cfg"].replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    return ReplayMotionsSceneCfg


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
        robot_type: str = "dex_evt",
        knee_modify: bool = False,
        start_frames: int = 0,
        end_frames: int = -1,
        correct_root_pose: bool = False,
        hold_pose_frames: int = 0,
    ):
        """Initialize motion loader with multi-stage interpolation pipeline.
        
        Pipeline stages:
        1. Load motion from file (convert quaternion format if needed)
        2. Add smooth start/end transitions with optional knee modification
        3. Resample to output fps using high-quality interpolation (SLERP for rotations)
        4. Compute velocities using numerical differentiation
        
        Args:
            motion_file: Path to npz/pkl motion file
            input_fps: Input motion frame rate
            output_fps: Desired output frame rate
            device: Torch device (cpu/cuda)
            frame_range: Optional (start, end) frame range to load
            robot_type: Robot type, only "dex_evt" is supported.
            knee_modify: Apply special knee interpolation for start/end transitions
            start_frames: Number of transition frames to add at start
            end_frames: Number of transition frames to add at end
        """
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self.robot_type = robot_type
        self.knee_modify = knee_modify
        self.correct_root_pose = correct_root_pose
        self.hold_pose_frames = max(0, hold_pose_frames)

        # 获取机器人完整配置
        self.robot_config = get_robot_config(robot_type)
        self.default_pose = self.robot_config["default_pose"]
        self._default_pose_tensor = torch.tensor(self.default_pose, dtype=torch.float32, device=self.device)

        print(f"[INFO] Robot type: {robot_type}, DOF: {self.robot_config['num_dof']}")
        self._load_motion()
        self._apply_root_pose_correction()
        self._interpolate_motion_startend(start_frames, end_frames)  # 首尾插值平滑
        self._interpolate_motion()
        self._append_hold_pose_frames()
        self._compute_velocities()

    def _load_motion(self):
        """Loads the motion from the npz/pkl file."""
        _ensure_numpy_core_compatibility()
        data = np.load(self.motion_file, allow_pickle=True)
        root_pos = torch.from_numpy(data['root_pos']).float().to(self.device)
        root_rot = torch.from_numpy(data['root_rot']).float().to(self.device)
        # convert root_rot from xyzw to wxyz
        root_rot = torch.cat([root_rot[:, 3:4], root_rot[:, :3]], dim=1)
        dof_pos = torch.from_numpy(data['dof_pos']).float().to(self.device)
        dof_pos = self._match_dof_layout(dof_pos)
        if self.frame_range is not None:
            start_frame, end_frame = self.frame_range
            root_pos = root_pos[start_frame-1:end_frame]
            root_rot = root_rot[start_frame-1:end_frame]
            dof_pos = dof_pos[start_frame-1:end_frame]
        self.motion_base_poss_input = root_pos
        self.motion_base_rots_input = root_rot
        self.motion_dof_poss_input = dof_pos
        self.input_frames = root_pos.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"Motion loaded ({self.motion_file}), duration: {self.duration} sec, frames: {self.input_frames}")

    def _match_dof_layout(self, dof_pos: torch.Tensor) -> torch.Tensor:
        """Match the loaded DOF layout to the target robot's layout by joint names."""
        target_joint_names = self.robot_config["joint_names"]
        target_dof = len(target_joint_names)
        if dof_pos.shape[1] == target_dof:
            return dof_pos
        input_dof = dof_pos.shape[1]
        source_joint_orders = self.robot_config.get("input_joint_orders", {})
        source_joint_order = source_joint_orders.get(input_dof, self.robot_config.get("input_joint_order"))
        if source_joint_order is None:
            raise ValueError(
                f"Input DOF count ({input_dof}) does not match target ({target_dof}) and no input_joint_order is "
                "specified for remapping."
            )
        if input_dof != len(source_joint_order):
            raise ValueError(
                f"Input DOF count ({input_dof}) does not match the provided input_joint_order "
                f"length ({len(source_joint_order)})."
            )
        name_to_idx = {name: idx for idx, name in enumerate(source_joint_order)}
        missing = [name for name in target_joint_names if name not in name_to_idx]
        if missing:
            raise ValueError(f"Cannot remap DOF layout; missing joints in source data: {missing}")
        ordered_indices = torch.tensor(
            [name_to_idx[name] for name in target_joint_names], dtype=torch.long, device=dof_pos.device
        )
        remapped = dof_pos.index_select(dim=1, index=ordered_indices)
        ignored = [name for name in source_joint_order if name not in target_joint_names]
        print(
            f"[INFO] Remapped DOF layout from {input_dof} to {target_dof} using joint name mapping."
        )
        if ignored:
            print(f"[INFO] Ignored source joints: {ignored}")
        return remapped

    def _apply_root_pose_correction(self):
        """Apply coupled pose correction if requested."""
        if not self.correct_root_pose:
            return
        base_pos = self.motion_base_poss_input.cpu().numpy()
        base_rot_wxyz = self.motion_base_rots_input.cpu().numpy()
        base_rot_xyzw = np.concatenate([base_rot_wxyz[:, 1:], base_rot_wxyz[:, 0:1]], axis=1)
        target_pos = np.array(self.robot_config.get("root_target_pos", [0.0, 0.0, 1.0]), dtype=np.float32)
        corrected_pos, corrected_rot_xyzw = correct_root_pose_coupled(base_pos, base_rot_xyzw, target_pos)
        corrected_rot_wxyz = np.concatenate([corrected_rot_xyzw[:, 3:], corrected_rot_xyzw[:, :3]], axis=1)
        self.motion_base_poss_input = torch.from_numpy(corrected_pos).float().to(self.device)
        self.motion_base_rots_input = torch.from_numpy(corrected_rot_wxyz).float().to(self.device)
        self.input_frames = self.motion_base_poss_input.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt

    def _interpolate_motion(self):
        """Interpolates the motion to the output fps."""
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        
        # 位置: 线性插值
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        
        # 旋转: SLERP插值 (保证四元数的连续性和单位长度)
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        
        # DOF: 线性插值
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        print(
            f"Motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps}, output frames:"
            f" {self.output_frames}, output fps: {self.output_fps}"
        )

    def _append_hold_pose_frames(self):
        """Optional tail segment that holds the default pose for a fixed number of frames."""
        if self.hold_pose_frames <= 0:
            return

        # Hold by simply repeating the last interpolated frame; no blending to default pose.
        hold_base_pos = self.motion_base_poss[-1:].repeat(self.hold_pose_frames, 1)
        hold_base_rot = self.motion_base_rots[-1:].repeat(self.hold_pose_frames, 1)
        hold_dof_pos = self.motion_dof_poss[-1:].repeat(self.hold_pose_frames, 1)

        self.motion_base_poss = torch.cat([self.motion_base_poss, hold_base_pos], dim=0)
        self.motion_base_rots = torch.cat([self.motion_base_rots, hold_base_rot], dim=0)
        self.motion_dof_poss = torch.cat([self.motion_dof_poss, hold_dof_pos], dim=0)
        self.output_frames = self.motion_base_poss.shape[0]
        self.duration = (self.output_frames - 1) * self.output_dt
        print(f"[INFO] Holding last frame for {self.hold_pose_frames} frames at the tail.")
    def _lower_dof_interpolation(self, start_dof, end_dof, nframe):
        """对下肢DOF进行特殊插值处理,膝关节先抬高再放下"""
        mid_frame = nframe // 2
        quat_frame = mid_frame // 2
        
        # 左腿下肢6个DOF: hip_pitch, hip_roll, hip_yaw, knee_pitch, ankle_pitch, ankle_roll
        left_lower_dof = np.linspace(start_dof[:6], end_dof[:6],
                        num=mid_frame+1,
                        endpoint=False)[1:].reshape(-1, 6)
        # 右腿下肢6个DOF
        right_lower_dof = np.linspace(start_dof[6:12], end_dof[6:12],
                        num=mid_frame+1,
                        endpoint=False)[1:].reshape(-1, 6)
        
        # 左膝关节特殊处理: 先弯曲到1.0,再恢复
        left_knee_1 = np.linspace(start_dof[3], 1.0, num=quat_frame).reshape(-1, 1)
        left_knee_2 = np.linspace(1.0, end_dof[3], num=quat_frame+1).reshape(-1, 1)
        
        # 右膝关节特殊处理
        right_knee_1 = np.linspace(start_dof[9], 1.0, num=quat_frame).reshape(-1, 1)
        right_knee_2 = np.linspace(1.0, end_dof[9], num=quat_frame+1).reshape(-1, 1)
        
        # 更新膝关节数据
        left_lower_dof[:, 3] = np.concatenate((left_knee_1, left_knee_2), axis=0).squeeze(-1)
        right_lower_dof[:, 3] = np.concatenate((right_knee_1, right_knee_2), axis=0).squeeze(-1)

        # 补齐到nframe长度
        left_lower_dof = np.concatenate((left_lower_dof, 
                                        np.tile(left_lower_dof[-1], (nframe-left_lower_dof.shape[0], 1))), 
                                       axis=0)
        right_lower_dof = np.concatenate((np.tile(right_lower_dof[0], (nframe-right_lower_dof.shape[0], 1)),
                                         right_lower_dof), 
                                        axis=0)
        
        lower_dof = np.concatenate((left_lower_dof, right_lower_dof), axis=1)
        return lower_dof

    def _interpolate_motion_startend(self, start_frame: int, end_frame: int):
        """改进的首尾插值,支持膝关节特殊处理和更精确的四元数插值
        
        注意: 内部统一使用wxyz格式的四元数,与Isaac Lab保持一致
        """
        import numpy as np

        # 使用机器人对应的默认姿态
        default_pose = np.array(self.default_pose)
        default_p = default_pose[0:3]           # root position
        default_r_wxyz = default_pose[3:7]      # root rotation (wxyz)
        default_dof = default_pose[7:]          # joint DOFs

        # 原始数据 (已经是wxyz格式)
        base_pos = self.motion_base_poss_input.cpu().numpy()
        base_rot_wxyz = self.motion_base_rots_input.cpu().numpy()  # wxyz格式
        dof_pos = self.motion_dof_poss_input.cpu().numpy()

        # 将wxyz转换为xyzw供scipy使用
        def wxyz_to_xyzw(q):
            return q[[1,2,3,0]]
        
        def xyzw_to_wxyz(q):
            return q[[3,0,1,2]]

        # 获取首尾帧的欧拉角 (ZYX顺序)
        start_rot_euler = R.from_quat(wxyz_to_xyzw(base_rot_wxyz[0])).as_euler('ZYX')
        end_rot_euler = R.from_quat(wxyz_to_xyzw(base_rot_wxyz[-1])).as_euler('ZYX')
        default_rot_euler = R.from_quat(wxyz_to_xyzw(default_r_wxyz)).as_euler('ZYX')

        # 首部插值 - 从默认姿态到第一帧
        if start_frame > 0:
            # Root位置: Z轴线性插值,XY保持第一帧
            start_z = np.linspace(default_p[2], base_pos[0, 2], start_frame)
            start_base_pos = np.zeros((start_frame, 3))
            start_base_pos[:, 0] = base_pos[0, 0]
            start_base_pos[:, 1] = base_pos[0, 1]
            start_base_pos[:, 2] = start_z
            
            # Root旋转: 保持Z轴(yaw)不变,只插值YX轴
            rotations_start = R.from_euler('ZYX', [
                np.concatenate((start_rot_euler[0:1], default_rot_euler[1:])),  # 保持第一帧Z轴
                np.concatenate((start_rot_euler[0:1], start_rot_euler[1:]))    # 目标姿态
            ])
            times = np.linspace(0, 1, start_frame)
            slerp = Slerp([0, 1], rotations_start)
            interp_rots = slerp(times).as_euler('ZYX')
            start_base_rot_xyzw = R.from_euler('ZYX', interp_rots).as_quat()  # xyzw格式
            start_base_rot_wxyz = np.array([xyzw_to_wxyz(q) for q in start_base_rot_xyzw])  # 转回wxyz
            
            # DOF插值
            if self.knee_modify and dof_pos.shape[1] >= 12:
                upper_dim = dof_pos.shape[1] - 12
                if upper_dim > 0:
                    upper_start_dof = np.linspace(
                        default_dof[12:],
                        dof_pos[0][12:],
                        num=start_frame + 1,
                        endpoint=False,
                    )[1:].reshape(-1, upper_dim)
                else:
                    upper_start_dof = np.zeros((start_frame, 0))
                # 下肢(legs): 特殊膝关节插值
                lower_start_dof = self._lower_dof_interpolation(default_dof[0:12], 
                                                                dof_pos[0][0:12], 
                                                                start_frame)
                start_dof_pos = np.concatenate((lower_start_dof, upper_start_dof), axis=1)
            else:
                # 简单线性插值
                start_dof_pos = np.linspace(default_dof, dof_pos[0],
                                          num=start_frame + 1,
                                          endpoint=False)[1:].reshape(-1, dof_pos.shape[1])
        else:
            start_base_pos = np.empty((0, 3))
            start_base_rot_wxyz = np.empty((0, 4))
            start_dof_pos = np.empty((0, dof_pos.shape[1]))

        # 尾部插值 - 从最后一帧到默认姿态
        if end_frame > 0:
            # Root位置: Z轴线性插值,XY保持最后一帧
            end_z = np.linspace(base_pos[-1, 2], default_p[2], end_frame + 1)[1:]
            end_base_pos = np.zeros((end_frame, 3))
            end_base_pos[:, 0] = base_pos[-1, 0]
            end_base_pos[:, 1] = base_pos[-1, 1]
            end_base_pos[:, 2] = end_z
            
            # Root旋转: 保持Z轴(yaw)不变,只插值YX轴
            rotations_end = R.from_euler('ZYX', [
                np.concatenate((end_rot_euler[0:1], default_rot_euler[1:])),
                np.concatenate((end_rot_euler[0:1], end_rot_euler[1:]))
            ])
            times = np.linspace(1, 0, end_frame)
            slerp = Slerp([0, 1], rotations_end)
            interp_rots = slerp(times).as_euler('ZYX')
            end_base_rot_xyzw = R.from_euler('ZYX', interp_rots).as_quat()  # xyzw格式
            end_base_rot_wxyz = np.array([xyzw_to_wxyz(q) for q in end_base_rot_xyzw])  # 转回wxyz
            
            # DOF插值
            if self.knee_modify and dof_pos.shape[1] >= 12:
                upper_dim = dof_pos.shape[1] - 12
                if upper_dim > 0:
                    upper_end_dof = np.linspace(
                        dof_pos[-1][12:],
                        default_dof[12:],
                        num=end_frame + 1,
                    )[1:].reshape(-1, upper_dim)
                else:
                    upper_end_dof = np.zeros((end_frame, 0))
                lower_end_dof = self._lower_dof_interpolation(dof_pos[-1][0:12], 
                                                              default_dof[0:12], 
                                                              end_frame)
                end_dof_pos = np.concatenate((lower_end_dof, upper_end_dof), axis=1)
            else:
                end_dof_pos = np.linspace(dof_pos[-1], default_dof,
                                        num=end_frame + 1)[1:].reshape(-1, dof_pos.shape[1])
        else:
            end_base_pos = np.empty((0, 3))
            end_base_rot_wxyz = np.empty((0, 4))
            end_dof_pos = np.empty((0, dof_pos.shape[1]))

        # 合并数据 (保持wxyz格式)
        new_base_pos = np.vstack([start_base_pos, base_pos, end_base_pos])
        new_base_rot_wxyz = np.vstack([start_base_rot_wxyz, base_rot_wxyz, end_base_rot_wxyz])
        new_dof_pos = np.vstack([start_dof_pos, dof_pos, end_dof_pos])

        # 转回torch (保持wxyz格式)
        self.motion_base_poss_input = torch.from_numpy(new_base_pos).float().to(self.device)
        self.motion_base_rots_input = torch.from_numpy(new_base_rot_wxyz).float().to(self.device)
        self.motion_dof_poss_input = torch.from_numpy(new_dof_pos).float().to(self.device)
        self.input_frames = new_base_pos.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        
        knee_mode = "with knee modification" if self.knee_modify else "linear"
        print(f"After start/end interpolation ({knee_mode}): "
              f"start={start_frame}, end={end_frame}, total frames={self.input_frames}")


    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between two tensors."""
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Spherical linear interpolation between two quaternions."""
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i].item())
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor):
        """Computes the frame blend for the motion."""
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1, device=self.device))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        """Computes the velocities of the motion using numerical differentiation."""
        # 线性速度: 使用torch.gradient计算位置的时间导数
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        
        # 关节速度: 使用torch.gradient计算DOF的时间导数
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        
        # 角速度: 使用SO3的正确导数计算
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        """Computes the derivative of a sequence of SO3 rotations.

        Args:
            rotations: shape (B, 4).
            dt: time step.
        Returns:
            shape (B, 3).
        """
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B−2, 4)

        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B−2, 3)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)  # repeat first and last sample
        return omega

    def get_next_state(self):
        """Gets the next state of the motion."""
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, robot_config: dict):
    """Runs the simulation loop.
    
    Args:
        sim: Simulation context
        scene: Interactive scene
        robot_config: Robot configuration dictionary
    """
    # Load motion
    motion = MotionLoader(
        motion_file=args_cli.input_file,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=args_cli.frame_range,
        robot_type=args_cli.robot,
        knee_modify=args_cli.knee_modify,
        start_frames=args_cli.start_frames,
        end_frames=args_cli.end_frames,
        correct_root_pose=args_cli.correct_root_pose_coupled,
        hold_pose_frames=args_cli.hold_pos,
    )

    # Extract scene entities
    robot = scene["robot"]
    joint_names = robot_config["joint_names"]
    height_offset = robot_config["height_offset"]
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

    # ------- data logger -------------------------------------------------------
    log = {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    file_saved = False
    # --------------------------------------------------------------------------

    # Simulation loop
    while simulation_app.is_running():
        state, reset_flag = motion.get_next_state()
        (
            motion_base_pos,
            motion_base_rot,
            motion_base_lin_vel,
            motion_base_ang_vel,
            motion_dof_pos,
            motion_dof_vel,
        ) = state
        # set root state
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, 2] += height_offset  # adjust height (机器人特定)
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        # set joint state
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos.to(torch.float32)
        joint_vel[:, robot_joint_indexes] = motion_dof_vel.to(torch.float32)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim.get_physics_dt())

        # pos_lookat = root_states[0, :3].cpu().numpy()
        # sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        if not file_saved:
            log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
            log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
            log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
            log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
            log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

        if reset_flag and not file_saved:
            file_saved = True
            for k in (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ):
                log[k] = np.stack(log[k], axis=0)

            output_dir = os.path.abspath(args_cli.output_dir)
            os.makedirs(output_dir, exist_ok=True)
            total_frames = log["joint_pos"].shape[0]
            split_frame = args_cli.split_frame

            def _save_motion(data_dict, suffix: str):
                path = os.path.join(output_dir, f"{args_cli.output_name}{suffix}.npz")
                np.savez(path, **data_dict)
                print(f"[INFO]: Motion saved to {path}")

            if split_frame is None or split_frame <= 0 or split_frame >= total_frames:
                _save_motion(log, "")
            else:
                head, tail = {}, {}
                for k, v in log.items():
                    if k == "fps":
                        head[k] = v
                        tail[k] = v
                    else:
                        head[k] = v[:split_frame]
                        tail[k] = v[split_frame:]

                _save_motion(head, f"_part1_{split_frame}")
                _save_motion(tail, f"_part2_{split_frame}")


def main():
    """Main function."""
    # 获取机器人配置
    robot_config = get_robot_config(args_cli.robot)
    print(f"[INFO]: Loading robot: {args_cli.robot}")
    print(f"[INFO]: Number of joints: {len(robot_config['joint_names'])}")
    
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    
    # Design scene - 根据robot_type动态创建场景配置
    scene_cfg_class = create_scene_cfg(args_cli.robot)
    scene_cfg = scene_cfg_class(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    
    # Run the simulator - 传递机器人配置
    run_simulator(sim, scene, robot_config)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
