from isaaclab.utils import configclass

from whole_body_tracking.robots.tiangong3 import D3_ACTION_SCALE, TIANGONG3_CFG
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class Tiangong3FlatWoStateEstimationEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Update the robot configuration
        self.scene.robot = TIANGONG3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = D3_ACTION_SCALE

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
