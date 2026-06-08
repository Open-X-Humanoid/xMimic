"""Convert MuJoCo-style qpos sequences to the training npz format by replaying them in Isaac Sim.

Input file requirements (npz):
    - qpos: shape (T, 7 + dof). Layout: [root_xyz, root_quat_wxyz, joint_angles...]
    - fps:  scalar frames-per-second (optional, can be overridden via CLI)

Output file (npz) keys match the tracking pipeline:
    fps, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w

Example:
    python qpos_to_npz_inter.py --input_file motion_data/omni_retarget/evt_climb_original.npz \\
        --output_name evt_climb_converted --robot dex_evt --output_dir motion_data/converted
"""

import argparse
import os
from typing import Callable

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from isaaclab.app import AppLauncher


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Convert qpos npz to training motion npz (via Isaac Sim replay).")
parser.add_argument("--input_file", type=str, required=True, help="Path to the input npz containing qpos.")
parser.add_argument(
    "--input_fps",
    type=int,
    default=None,
    help="Override the fps stored in the npz (defaults to the file value if present).",
)
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help="Optional 1-based inclusive frame range to crop from the input.",
)
parser.add_argument(
    "--pre_frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help="Optional 1-based inclusive frame range to crop before any corrections.",
)
parser.add_argument("--output_name", type=str, required=True, help="Name of the output motion npz (without suffix).")
parser.add_argument(
    "--output_dir",
    type=str,
    default="motion_data",
    help="Directory where the converted motion npz file will be stored.",
)
parser.add_argument(
    "--output_fps",
    type=int,
    default=None,
    help="Target fps of the output (defaults to the resolved input fps).",
)
parser.add_argument(
    "--start_frames",
    type=int,
    default=0,
    help="Optional frames to prepend by interpolating from default pose to the first frame.",
)
parser.add_argument(
    "--end_frames",
    type=int,
    default=0,
    help="Optional frames to append by interpolating from the last frame to the default pose.",
)
parser.add_argument(
    "--knee_modify",
    action="store_true",
    help="Enable special knee interpolation for start/end transitions (if robot legs have >=12 DOFs).",
)
parser.add_argument(
    "--hold_pos",
    type=int,
    default=0,
    help="Optional number of frames to hold the last pose at the tail (after interpolation).",
)
parser.add_argument(
    "--robot",
    type=str,
    default="dex_evt",
    choices=["dex_evt"],
    help="Target robot to replay and log. This repository currently ships only the dex_evt robot config.",
)
parser.add_argument(
    "--correct_root_pose_coupled",
    action="store_true",
    help="Apply coupled root pose correction (position + yaw) before interpolation.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# Delay all other Isaac imports until after SimulationApp is created (Carbonite requirement)
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp


from whole_body_tracking.robots.dex_evt import DEX_EVT_CFG


ROBOT_CONFIG_TIANGONG = {
    "default_pose": [
        0.0,
        0.0,
        -0.0,  # root position (x, y, z)
        1.0,
        0.0,
        0.0,
        0.0,  # root rotation (w, x, y, z)
        -0.25,
        0.0,
        0.0,
        0.5,
        -0.25,
        0.0,  # left leg
        -0.25,
        0.0,
        0.0,
        0.5,
        -0.25,
        0.0,  # right leg
        0.0,
        0.0,
        0.0,  # waist
        0.0,
        0.3,
        0.0,
        -0.3,
        0.0,
        0.0,
        0.0,  # left arm
        0.0,
        -0.3,
        0.0,
        -0.3,
        0.0,
        0.0,
        0.0,  # right arm
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
        "elbow_yaw_l_joint",
        "wrist_pitch_l_joint",
        "wrist_roll_l_joint",
        "shoulder_pitch_r_joint",
        "shoulder_roll_r_joint",
        "shoulder_yaw_r_joint",
        "elbow_pitch_r_joint",
        "elbow_yaw_r_joint",
        "wrist_pitch_r_joint",
        "wrist_roll_r_joint",
    ],
    "isaac_lab_cfg": None,
    "num_dof": 29,
    "height_offset": 0.96,
    "input_joint_order": None,
}

ROBOT_CONFIG_G1 = {
    "default_pose": [
        0.0,
        0.0,
        1.0,  # root position (x, y, z)
        1.0,
        0.0,
        0.0,
        0.0,  # root rotation (w, x, y, z)
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,  # left leg
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,  # right leg
        0.0,
        0.0,
        0.0,  # waist
        0.0,
        0.3,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,  # left arm
        0.0,
        -0.3,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,  # right arm
    ],
    "joint_names": [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
    "isaac_lab_cfg": None,
    "num_dof": 29,
    "height_offset": 0.0,
    "input_joint_order": None,
}

ROBOT_CONFIG_DEX_V3 = {
    "default_pose": [
        0.0,
        0.0,
        1.0,  # root position
        1.0,
        0.0,
        0.0,
        0.0,  # root rotation (wxyz)
        # left leg
        -0.25,
        0.0,
        0.0,
        0.5,
        -0.25,
        0.0,
        # right leg
        -0.25,
        0.0,
        0.0,
        0.5,
        -0.25,
        0.0,
        # waist
        0.0,
        0.0,
        0.0,
        # left arm
        0.0,
        0.3,
        0.0,
        -0.3,
        # right arm
        0.0,
        -0.3,
        0.0,
        -0.3,
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
    "isaac_lab_cfg": None,
    "num_dof": 23,
    "height_offset": 0.0,
    "input_joint_order": None,
}

ROBOT_CONFIG_DEX_EVT = {
    "default_pose": [
        0.0,
        0.0,
        0.95,  # root position
        0.70710678,
        0.0,
        0.0,
        0.70710678,  # root rotation (wxyz)
        # left leg
        -0.25,
        0.0,
        0.0,
        0.5,
        -0.25,
        0.0,
        # right leg
        -0.25,
        0.0,
        0.0,
        0.5,
        -0.25,
        0.0,
        # waist
        0.0,
        0.0,
        0.0,
        # left arm
        0.0,
        0.3,
        0.0,
        -0.3,
        # 0.0,
        # 0.0,
        # 0.0,
        # right arm
        0.0,
        -0.3,
        0.0,
        -0.3,
        # 0.0,
        # 0.0,
        # 0.0,
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
        # "elbow_yaw_l_joint", 
        # "wrist_pitch_l_joint",
        # "wrist_roll_l_joint",
        "shoulder_pitch_r_joint",
        "shoulder_roll_r_joint",
        "shoulder_yaw_r_joint",
        "elbow_pitch_r_joint",
        # "elbow_yaw_r_joint", 
        # "wrist_pitch_r_joint",
        # "wrist_roll_r_joint",
    ],
    "isaac_lab_cfg": DEX_EVT_CFG,
    "num_dof": 23,
    "height_offset": 0.0,
    "input_joint_order": None,
}

# Populate default input joint orders for remapping from 29-DOF layouts when needed.
ROBOT_CONFIG_TIANGONG["input_joint_order"] = list(ROBOT_CONFIG_TIANGONG["joint_names"])
ROBOT_CONFIG_G1["input_joint_order"] = list(ROBOT_CONFIG_G1["joint_names"])
ROBOT_CONFIG_DEX_V3["input_joint_order"] = list(ROBOT_CONFIG_TIANGONG["joint_names"])
ROBOT_CONFIG_DEX_EVT["input_joint_order"] = list(ROBOT_CONFIG_TIANGONG["joint_names"])

ROBOT_CONFIGS = {
    "dex_evt": ROBOT_CONFIG_DEX_EVT,
}


def correct_root_pose_coupled(root_pos: np.ndarray, root_rot_xyzw: np.ndarray, target_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coupled translation + yaw correction for the root pose (keeps original z)."""
    root_pos = np.asarray(root_pos, dtype=np.float64)
    root_rot_xyzw = np.asarray(root_rot_xyzw, dtype=np.float64)
    target_pos = np.asarray(target_pos, dtype=np.float64)

    initial_pos = root_pos[0].copy()
    initial_quat_xyzw = root_rot_xyzw[0].copy()
    initial_rot = R.from_quat(initial_quat_xyzw)

    target_rot = R.identity()
    initial_euler = initial_rot.as_euler("zyx", degrees=False)
    target_euler = target_rot.as_euler("zyx", degrees=False)
    yaw_correction = target_euler[0] - initial_euler[0]

    correction_rot = R.from_euler("z", yaw_correction)
    correction_T = np.eye(4, dtype=np.float64)
    correction_T[:3, :3] = correction_rot.as_matrix()
    target_pos_xy = target_pos.copy()
    # target_pos_xy[2] = initial_pos[2]  # keep z unchanged
    correction_T[:3, 3] = target_pos_xy - correction_rot.apply(initial_pos)

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
    return corrected_positions, corrected_rotations


def get_robot_config(robot_type: str) -> dict:
    if robot_type not in ROBOT_CONFIGS:
        raise ValueError(f"Unsupported robot type '{robot_type}'. Options: {list(ROBOT_CONFIGS.keys())}")
    return ROBOT_CONFIGS[robot_type]


def create_scene_cfg(robot_type: str) -> Callable[..., InteractiveSceneCfg]:
    """Return a scene config bound to the chosen robot."""
    robot_config = get_robot_config(robot_type)

    @configclass
    class ReplayMotionsSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                intensity=750.0,
                texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            ),
        )
        robot: ArticulationCfg = robot_config["isaac_lab_cfg"].replace(prim_path="{ENV_REGEX_NS}/Robot")

    return ReplayMotionsSceneCfg


# -----------------------------------------------------------------------------
# Motion loader
# -----------------------------------------------------------------------------
class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
        pre_frame_range: tuple[int, int] | None,
        correct_root_pose: bool,
        robot_config: dict,
        start_frames: int = 0,
        end_frames: int = 0,
        knee_modify: bool = False,
        hold_pose_frames: int = 0,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.device = device
        self.frame_range = frame_range
        self.pre_frame_range = pre_frame_range
        self.correct_root_pose = correct_root_pose
        self.robot_config = robot_config
        self.current_idx = 0
        self.start_frames = max(0, start_frames)
        self.end_frames = max(0, end_frames)
        self.knee_modify = knee_modify
        self.hold_pose_frames = max(0, hold_pose_frames)
        self.default_pose = np.array(self.robot_config["default_pose"], dtype=np.float32)

        self._load_motion()
        self._apply_pre_frame_range()
        self._apply_root_pose_correction()
        self._apply_frame_range()
        self._interpolate_motion_startend(self.start_frames, self.end_frames)
        self._interpolate_motion()
        self._append_hold_pose_frames()
        self._compute_velocities()

    def _load_motion(self):
        """Load qpos array and split into root pose + dof positions."""
        data = np.load(self.motion_file, allow_pickle=True)
        qpos = data["qpos"]
        root_pos = torch.from_numpy(qpos[:, :3]).float().to(self.device)
        root_rot = torch.from_numpy(qpos[:, 3:7]).float().to(self.device)  # wxyz already
        dof_pos = torch.from_numpy(qpos[:, 7:]).float().to(self.device)

        dof_pos = self._match_dof_layout(dof_pos)

        self.motion_base_poss_input = root_pos
        self.motion_base_rots_input = root_rot
        self.motion_dof_poss_input = dof_pos
        self._update_duration()
        print(
            f"Motion loaded ({self.motion_file}), duration: {self.duration:.3f}s, "
            f"frames: {self.input_frames}, input fps: {self.input_fps}"
        )

    def _update_duration(self):
        self.input_frames = self.motion_base_poss_input.shape[0]
        self.duration = max((self.input_frames - 1) * self.input_dt, self.input_dt)

    def _apply_pre_frame_range(self):
        """Optional cropping before any corrections."""
        if self.pre_frame_range is None:
            return
        start_frame, end_frame = self.pre_frame_range
        self.motion_base_poss_input = self.motion_base_poss_input[start_frame - 1 : end_frame]
        self.motion_base_rots_input = self.motion_base_rots_input[start_frame - 1 : end_frame]
        self.motion_dof_poss_input = self.motion_dof_poss_input[start_frame - 1 : end_frame]
        self._update_duration()
        print(f"[INFO] Applied pre_frame_range {self.pre_frame_range}, frames now: {self.input_frames}")

    def _apply_root_pose_correction(self):
        """Apply coupled root pose correction if enabled."""
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
        self._update_duration()

    def _apply_frame_range(self):
        """Crop after any corrections."""
        if self.frame_range is None:
            return
        start_frame, end_frame = self.frame_range
        self.motion_base_poss_input = self.motion_base_poss_input[start_frame - 1 : end_frame]
        self.motion_base_rots_input = self.motion_base_rots_input[start_frame - 1 : end_frame]
        self.motion_dof_poss_input = self.motion_dof_poss_input[start_frame - 1 : end_frame]
        self._update_duration()
        print(f"[INFO] Applied frame_range {self.frame_range}, frames now: {self.input_frames}")

    def _match_dof_layout(self, dof_pos: torch.Tensor) -> torch.Tensor:
        target_dof = self.robot_config["num_dof"]
        if dof_pos.shape[1] == target_dof:
            return dof_pos

        source_joint_order = self.robot_config.get("input_joint_order")
        target_joint_names = self.robot_config["joint_names"]
        if source_joint_order is None:
            raise ValueError(
                f"Input DOF ({dof_pos.shape[1]}) does not match target ({target_dof}) "
                "and no input_joint_order is provided for remapping."
            )
        if dof_pos.shape[1] != len(source_joint_order):
            raise ValueError(
                f"Input DOF ({dof_pos.shape[1]}) does not match the provided input_joint_order "
                f"length ({len(source_joint_order)})."
            )
        name_to_idx = {name: idx for idx, name in enumerate(source_joint_order)}
        ordered_indices = torch.tensor(
            [name_to_idx[name] for name in target_joint_names], dtype=torch.long, device=dof_pos.device
        )
        remapped = dof_pos.index_select(dim=1, index=ordered_indices)
        print(f"Remapped DOF from {dof_pos.shape[1]} -> {target_dof} using input_joint_order.")
        return remapped

    def _interpolate_motion(self):
        """Resample motion to the desired output fps."""
        if self.input_frames == 1:
            self.motion_base_poss = self.motion_base_poss_input
            self.motion_base_rots = self.motion_base_rots_input
            self.motion_dof_poss = self.motion_dof_poss_input
            self.output_frames = 1
            return

        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        print(
            f"Interpolated: input frames {self.input_frames} @ {self.input_fps} fps -> "
            f"{self.output_frames} frames @ {self.output_fps} fps"
        )

    def _interpolate_motion_startend(self, start_frame: int, end_frame: int):
        """Smoothly transition between default pose and motion at the head/tail."""
        if start_frame <= 0 and end_frame <= 0:
            return

        default_p = self.default_pose[0:3]
        default_r_wxyz = self.default_pose[3:7]
        default_dof = self.default_pose[7:]

        base_pos = self.motion_base_poss_input.cpu().numpy()
        base_rot_wxyz = self.motion_base_rots_input.cpu().numpy()
        dof_pos = self.motion_dof_poss_input.cpu().numpy()

        def wxyz_to_xyzw(q):
            return q[[1, 2, 3, 0]]

        def xyzw_to_wxyz(q):
            return q[[3, 0, 1, 2]]

        start_rot_euler = R.from_quat(wxyz_to_xyzw(base_rot_wxyz[0])).as_euler("ZYX")
        end_rot_euler = R.from_quat(wxyz_to_xyzw(base_rot_wxyz[-1])).as_euler("ZYX")
        default_rot_euler = R.from_quat(wxyz_to_xyzw(default_r_wxyz)).as_euler("ZYX")

        if start_frame > 0:
            start_z = np.linspace(default_p[2], base_pos[0, 2], start_frame)
            start_base_pos = np.zeros((start_frame, 3))
            start_base_pos[:, 0] = base_pos[0, 0]
            start_base_pos[:, 1] = base_pos[0, 1]
            start_base_pos[:, 2] = start_z

            rotations_start = R.from_euler(
                "ZYX",
                [
                    np.concatenate((start_rot_euler[0:1], default_rot_euler[1:])),
                    np.concatenate((start_rot_euler[0:1], start_rot_euler[1:])),
                ],
            )
            times = np.linspace(0, 1, start_frame)
            slerp = Slerp([0, 1], rotations_start)
            interp_rots = slerp(times).as_euler("ZYX")
            start_base_rot_xyzw = R.from_euler("ZYX", interp_rots).as_quat()
            start_base_rot_wxyz = np.array([xyzw_to_wxyz(q) for q in start_base_rot_xyzw])

            if self.knee_modify and dof_pos.shape[1] >= 12:
                upper_dim = dof_pos.shape[1] - 12
                upper_start_dof = (
                    np.linspace(default_dof[12:], dof_pos[0][12:], num=start_frame + 1, endpoint=False)[1:].reshape(
                        -1, upper_dim
                    )
                    if upper_dim > 0
                    else np.zeros((start_frame, 0))
                )
                lower_start_dof = self._lower_dof_interpolation(default_dof[0:12], dof_pos[0][0:12], start_frame)
                start_dof_pos = np.concatenate((lower_start_dof, upper_start_dof), axis=1)
            else:
                start_dof_pos = (
                    np.linspace(default_dof, dof_pos[0], num=start_frame + 1, endpoint=False)[1:].reshape(
                        -1, dof_pos.shape[1]
                    )
                )
        else:
            start_base_pos = np.empty((0, 3))
            start_base_rot_wxyz = np.empty((0, 4))
            start_dof_pos = np.empty((0, dof_pos.shape[1]))

        if end_frame > 0:
            end_z = np.linspace(base_pos[-1, 2], default_p[2], end_frame + 1)[1:]
            end_base_pos = np.zeros((end_frame, 3))
            end_base_pos[:, 0] = base_pos[-1, 0]
            end_base_pos[:, 1] = base_pos[-1, 1]
            end_base_pos[:, 2] = end_z

            rotations_end = R.from_euler(
                "ZYX",
                [
                    np.concatenate((end_rot_euler[0:1], default_rot_euler[1:])),
                    np.concatenate((end_rot_euler[0:1], end_rot_euler[1:])),
                ],
            )
            times = np.linspace(1, 0, end_frame)
            slerp = Slerp([0, 1], rotations_end)
            interp_rots = slerp(times).as_euler("ZYX")
            end_base_rot_xyzw = R.from_euler("ZYX", interp_rots).as_quat()
            end_base_rot_wxyz = np.array([xyzw_to_wxyz(q) for q in end_base_rot_xyzw])

            if self.knee_modify and dof_pos.shape[1] >= 12:
                upper_dim = dof_pos.shape[1] - 12
                upper_end_dof = (
                    np.linspace(dof_pos[-1][12:], default_dof[12:], num=end_frame + 1)[1:].reshape(-1, upper_dim)
                    if upper_dim > 0
                    else np.zeros((end_frame, 0))
                )
                lower_end_dof = self._lower_dof_interpolation(dof_pos[-1][0:12], default_dof[0:12], end_frame)
                end_dof_pos = np.concatenate((lower_end_dof, upper_end_dof), axis=1)
            else:
                end_dof_pos = np.linspace(dof_pos[-1], default_dof, num=end_frame + 1)[1:].reshape(
                    -1, dof_pos.shape[1]
                )
        else:
            end_base_pos = np.empty((0, 3))
            end_base_rot_wxyz = np.empty((0, 4))
            end_dof_pos = np.empty((0, dof_pos.shape[1]))

        new_base_pos = np.vstack([start_base_pos, base_pos, end_base_pos])
        new_base_rot_wxyz = np.vstack([start_base_rot_wxyz, base_rot_wxyz, end_base_rot_wxyz])
        new_dof_pos = np.vstack([start_dof_pos, dof_pos, end_dof_pos])

        self.motion_base_poss_input = torch.from_numpy(new_base_pos).float().to(self.device)
        self.motion_base_rots_input = torch.from_numpy(new_base_rot_wxyz).float().to(self.device)
        self.motion_dof_poss_input = torch.from_numpy(new_dof_pos).float().to(self.device)
        self.input_frames = new_base_pos.shape[0]
        self.duration = max((self.input_frames - 1) * self.input_dt, self.input_dt)
        knee_mode = "with knee modification" if self.knee_modify else "linear"
        print(
            f"[INFO] Start/end interpolation ({knee_mode}): start={start_frame}, end={end_frame}, "
            f"total frames={self.input_frames}"
        )

    def _lower_dof_interpolation(self, start_dof: np.ndarray, end_dof: np.ndarray, nframe: int) -> np.ndarray:
        """Special knee-up-then-down interpolation for leg DOFs (first 12 entries)."""
        mid_frame = nframe // 2
        quat_frame = max(1, mid_frame // 2)

        if mid_frame > 0:
            left_lower_dof = np.linspace(start_dof[:6], end_dof[:6], num=mid_frame + 1, endpoint=False)[1:].reshape(
                -1, 6
            )
            right_lower_dof = np.linspace(
                start_dof[6:12], end_dof[6:12], num=mid_frame + 1, endpoint=False
            )[1:].reshape(-1, 6)
        else:
            left_lower_dof = np.empty((0, 6))
            right_lower_dof = np.empty((0, 6))

        left_knee_1 = np.linspace(start_dof[3], 1.0, num=quat_frame).reshape(-1, 1)
        left_knee_2 = np.linspace(1.0, end_dof[3], num=quat_frame + 1).reshape(-1, 1)
        right_knee_1 = np.linspace(start_dof[9], 1.0, num=quat_frame).reshape(-1, 1)
        right_knee_2 = np.linspace(1.0, end_dof[9], num=quat_frame + 1).reshape(-1, 1)

        if left_lower_dof.shape[0] > 0:
            left_lower_dof[: quat_frame, 3:4] = left_knee_1
            left_lower_dof[quat_frame:, 3:4] = left_knee_2[1:]
        if right_lower_dof.shape[0] > 0:
            right_lower_dof[: quat_frame, 3:4] = right_knee_1
            right_lower_dof[quat_frame:, 3:4] = right_knee_2[1:]

        return np.concatenate((left_lower_dof, right_lower_dof), axis=1)

    def _append_hold_pose_frames(self):
        """Optionally repeat the last interpolated frame for a fixed number of frames."""
        if self.hold_pose_frames <= 0:
            return

        hold_base_pos = self.motion_base_poss[-1:].repeat(self.hold_pose_frames, 1)
        hold_base_rot = self.motion_base_rots[-1:].repeat(self.hold_pose_frames, 1)
        hold_dof_pos = self.motion_dof_poss[-1:].repeat(self.hold_pose_frames, 1)

        self.motion_base_poss = torch.cat([self.motion_base_poss, hold_base_pos], dim=0)
        self.motion_base_rots = torch.cat([self.motion_base_rots, hold_base_rot], dim=0)
        self.motion_dof_poss = torch.cat([self.motion_dof_poss, hold_dof_pos], dim=0)
        self.output_frames = self.motion_base_poss.shape[0]
        self.duration = (self.output_frames - 1) * self.output_dt
        print(f"[INFO] Holding last frame for {self.hold_pose_frames} frames at the tail.")

    def _compute_frame_blend(self, times: torch.Tensor):
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1, device=self.device))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], float(blend[i].item()))
        return slerped_quats

    def _compute_velocities(self):
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        if rotations.shape[0] < 3:
            return torch.zeros(rotations.shape[0], 3, device=rotations.device, dtype=rotations.dtype)
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
        return omega

    def get_next_state(self):
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


# -----------------------------------------------------------------------------
# Simulator loop
# -----------------------------------------------------------------------------
def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    robot_config: dict,
    input_fps: int,
    output_fps: int,
    pre_frame_range: tuple[int, int] | None,
    correct_root_pose: bool,
    start_frames: int,
    end_frames: int,
    knee_modify: bool,
    hold_pose_frames: int,
) -> None:
    motion = MotionLoader(
        motion_file=args_cli.input_file,
        input_fps=input_fps,
        output_fps=output_fps,
        device=sim.device,
        frame_range=args_cli.frame_range,
        pre_frame_range=pre_frame_range,
        correct_root_pose=correct_root_pose,
        robot_config=robot_config,
        start_frames=start_frames,
        end_frames=end_frames,
        knee_modify=knee_modify,
        hold_pose_frames=hold_pose_frames,
    )

    robot = scene["robot"]
    joint_names = robot_config["joint_names"]
    height_offset = robot_config["height_offset"]
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

    log = {
        "fps": [output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    file_saved = False

    while simulation_app.is_running():
        (
            motion_base_pos,
            motion_base_rot,
            motion_base_lin_vel,
            motion_base_ang_vel,
            motion_dof_pos,
            motion_dof_vel,
        ), reset_flag = motion.get_next_state()

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, 2] += height_offset
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos.to(torch.float32)
        joint_vel[:, robot_joint_indexes] = motion_dof_vel.to(torch.float32)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.render()  # We don't want physics integration (sim.step()).
        scene.update(sim.get_physics_dt())

        if not file_saved:
            log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
            log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
            log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
            log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
            log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

        if reset_flag and not file_saved:
            file_saved = True
            for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
                log[k] = np.stack(log[k], axis=0)

            output_dir = os.path.abspath(args_cli.output_dir)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{args_cli.output_name}.npz")
            np.savez(output_path, **log)
            print(f"[INFO] Motion saved to {output_path}")


def _resolve_fps(input_file: str, input_fps_arg: int | None, output_fps_arg: int | None) -> tuple[int, int]:
    """Determine input/output fps with sensible fallbacks."""
    file_fps = None
    with np.load(input_file, allow_pickle=True) as data:
        if "fps" in data:
            file_fps = int(np.array(data["fps"]).reshape(-1)[0])

    input_fps = input_fps_arg or file_fps or 30
    output_fps = output_fps_arg or input_fps
    return input_fps, output_fps


def main():
    robot_config = get_robot_config(args_cli.robot)
    input_fps, output_fps = _resolve_fps(args_cli.input_file, args_cli.input_fps, args_cli.output_fps)
    print(f"[INFO] Robot: {args_cli.robot}, input_fps: {input_fps}, output_fps: {output_fps}")

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / output_fps
    sim = SimulationContext(sim_cfg)

    SceneCfg = create_scene_cfg(args_cli.robot)
    scene_cfg = SceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    run_simulator(
        sim,
        scene,
        robot_config,
        input_fps=input_fps,
        output_fps=output_fps,
        pre_frame_range=args_cli.pre_frame_range,
        correct_root_pose=args_cli.correct_root_pose_coupled,
        start_frames=args_cli.start_frames,
        end_frames=args_cli.end_frames,
        knee_modify=args_cli.knee_modify,
        hold_pose_frames=args_cli.hold_pos,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
