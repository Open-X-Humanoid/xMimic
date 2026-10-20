from isaaclab.utils import configclass

from whole_body_tracking.robots.tiangong3 import (
    TIANGONG3_23DOF_ACTION_SCALE,
    TIANGONG3_23DOF_CFG,
    TIANGONG3_29DOF_ACTION_SCALE,
    TIANGONG3_29DOF_CFG,
)
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class Tiangong3Flat23DoFEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Update the robot configuration
        self.scene.robot = TIANGONG3_23DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = TIANGONG3_23DOF_ACTION_SCALE

        # Set the anchor body for motion commands
        self.commands.motion.anchor_body = "pelvis"

        # Define the body names based on the URDF structure
        self.commands.motion.body_names = [
            "pelvis",
            "hip_pitch_l_link",
            "hip_roll_l_link",
            "hip_yaw_l_link",
            "knee_pitch_l_link",
            "ankle_pitch_l_link",
            "ankle_roll_l_link",
            "hip_pitch_r_link",
            "hip_roll_r_link",
            "hip_yaw_r_link",
            "knee_pitch_r_link",
            "ankle_pitch_r_link",
            "ankle_roll_r_link",
            "waist_yaw_link",
            "waist_roll_link",
            "waist_pitch_link",
            "shoulder_pitch_l_link",
            "shoulder_roll_l_link",
            "shoulder_yaw_l_link",
            "elbow_pitch_l_link",
            # "elbow_yaw_l_link",
            # "wrist_pitch_l_link",
            # "wrist_roll_l_link",
            "shoulder_pitch_r_link",
            "shoulder_roll_r_link",
            "shoulder_yaw_r_link",
            "elbow_pitch_r_link",
            # "elbow_yaw_r_link",
            # "wrist_pitch_r_link",
            # "wrist_roll_r_link"
        ]
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class Tiangong3Flat29DoFEnvCfg(Tiangong3Flat23DoFEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = TIANGONG3_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = TIANGONG3_29DOF_ACTION_SCALE

        # Track the six newly articulated links in arm-chain order.
        for side in ("l", "r"):
            index = self.commands.motion.body_names.index(f"elbow_pitch_{side}_link") + 1
            self.commands.motion.body_names[index:index] = [
                f"elbow_yaw_{side}_link",
                f"wrist_pitch_{side}_link",
                f"wrist_roll_{side}_link",
            ]
