import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR
from whole_body_tracking.robots.actuator import DelayedImplicitActuatorCfg


DEX_EVT_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/dex_evt/urdf/tiangong2dex.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.95),
        joint_pos={
            "hip_pitch_l_joint": -0.25,   # l_hip_pitch
            "hip_roll_l_joint": 0.0,    # l_hip_roll
            "hip_yaw_l_joint": 0.0,   # l_hip_yaw
            "knee_pitch_l_joint": 0.5,  # l_knee
            "ankle_pitch_l_joint": -0.25, # l_ankle_pitch
            "ankle_roll_l_joint": 0.0,  # l_ankle_roll
            "hip_pitch_r_joint": -0.25,   # r_hip_pitch
            "hip_roll_r_joint": 0.0,    # r_hip_roll
            "hip_yaw_r_joint": 0.0,   # r_hip_yaw
            "knee_pitch_r_joint": 0.5,  # r_knee
            "ankle_pitch_r_joint": -0.25, # r_ankle_pitch
            "ankle_roll_r_joint": 0.0,  # r_ankle_roll
            "waist_yaw_joint": 0.0,     # waist_yaw
            "waist_roll_joint": 0.0,    # waist_roll
            "waist_pitch_joint": 0.0,   # waist_pitch
            "shoulder_pitch_l_joint": 0.0, # l_shoulder_pitch
            "shoulder_roll_l_joint": 0.3,  # l_shoulder_roll
            "shoulder_yaw_l_joint": 0.0,   # l_shoulder_yaw
            "elbow_pitch_l_joint": -0.3,   # l_elbow
            # "elbow_yaw_l_joint": 0.0,      # l_wrist_yaw
            # "wrist_pitch_l_joint": 0.0,    # l_wrist_pitch
            # "wrist_roll_l_joint": 0.0,     # l_wrist_roll
            "shoulder_pitch_r_joint": 0.0, # r_shoulder_pitch
            "shoulder_roll_r_joint": -0.3, # r_shoulder_roll
            "shoulder_yaw_r_joint": 0.0,   # r_shoulder_yaw
            "elbow_pitch_r_joint": -0.3,   # r_elbow
            # "elbow_yaw_r_joint": 0.0,      # r_wrist_yaw
            # "wrist_pitch_r_joint": 0.0,    # r_wrist_pitch
            # "wrist_roll_r_joint": 0.0     # r_wrist_roll
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators = {
        "legs": DelayedImplicitActuatorCfg(
            min_delay=0,
            max_delay=4,
            joint_names_expr=[
                ".*hip_yaw.*joint",
                ".*hip_roll.*joint",
                ".*hip_pitch.*joint",
                ".*knee_pitch.*joint",
            ],
            effort_limit_sim={
                ".*hip_yaw.*joint": 150,
                ".*hip_roll.*joint": 235,
                ".*hip_pitch.*joint": 235,
                ".*knee_pitch.*joint": 400,
            },
            velocity_limit_sim={
                ".*hip_yaw.*joint": 13.82300767579509,
                ".*hip_roll.*joint": 16.755160819145562,
                ".*hip_pitch.*joint": 16.755160819145562,
                ".*knee_pitch.*joint": 11.100294042683936,
            },
            stiffness={
                ".*hip_yaw.*joint": 150,
                ".*hip_roll.*joint": 300,
                ".*hip_pitch.*joint": 300,
                ".*knee_pitch.*joint": 350,
            },  # <-- 你来填
            damping={
                # ".*hip_yaw.*joint": 5,
                # ".*hip_roll.*joint": 10,
                # ".*hip_pitch.*joint": 10,
                # ".*knee_pitch.*joint": 10,
                ".*hip_yaw.*joint": 7.5,
                ".*hip_roll.*joint": 15,
                ".*hip_pitch.*joint": 15,
                ".*knee_pitch.*joint": 15,
            },    # <-- 你来填
        ),

        "feet": DelayedImplicitActuatorCfg(
            min_delay=0,
            max_delay=4,
            joint_names_expr=[
                ".*ankle_pitch.*joint",
                ".*ankle_roll.*joint",
            ],
            effort_limit_sim={
                ".*ankle_pitch.*joint": 55,
                ".*ankle_roll.*joint": 55,
            },
            velocity_limit_sim={
                ".*ankle_pitch.*joint": 14.137166941154069,
                ".*ankle_roll.*joint": 14.137166941154069,
            },
            stiffness={
                ".*ankle_pitch.*joint": 30,
                ".*ankle_roll.*joint": 16.8,
            },  # <-- 你来填
            damping={
                # ".*ankle_pitch.*joint": 1.4,
                # ".*ankle_roll.*joint": 1.4,
                ".*ankle_pitch.*joint": 3.75,
                ".*ankle_roll.*joint": 2.1,
            },    # <-- 你来填
        ),

        "waist": DelayedImplicitActuatorCfg(
            min_delay=0,
            max_delay=4,
            joint_names_expr=[
                "waist_yaw_joint", 
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            effort_limit_sim={
                "waist_yaw_joint": 91,
                "waist_roll_joint": 91,
                "waist_pitch_joint": 91,
            },
            velocity_limit_sim={
                "waist_yaw_joint": 9.42477796076938,
                "waist_roll_joint": 9.42477796076938,
                "waist_pitch_joint": 9.42477796076938,
            },
            stiffness={
                "waist_yaw_joint": 400,
                "waist_roll_joint": 400,
                "waist_pitch_joint": 400,
            },  # <-- 你来填
            damping={
                # "waist_yaw_joint": 5,
                # "waist_roll_joint": 10,
                # "waist_pitch_joint": 10,
                "waist_yaw_joint": 7.5,
                "waist_roll_joint": 15,
                "waist_pitch_joint": 15,
            },    # <-- 你来填
        ),

        "arms": DelayedImplicitActuatorCfg(
            min_delay=0,
            max_delay=4,
            joint_names_expr=[
                ".*shoulder_pitch.*joint",
                ".*shoulder_roll.*joint",
                ".*shoulder_yaw.*joint",
                ".*elbow_pitch.*joint",
                # ".*elbow_yaw.*joint",
                # ".*wrist_roll.*joint",
                # ".*wrist_pitch.*joint",
            ],
            effort_limit_sim={
                ".*shoulder_pitch.*joint": 90,
                ".*shoulder_roll.*joint": 90,
                ".*shoulder_yaw.*joint": 50,
                ".*elbow_pitch.*joint": 50,
                # ".*elbow_yaw.*joint": 36,
                # ".*wrist_roll.*joint": 36,
                # ".*wrist_pitch.*joint": 36,
            },
            velocity_limit_sim={
                ".*shoulder_pitch.*joint": 7.218332720398149,
                ".*shoulder_roll.*joint": 7.218332720398149,
                ".*shoulder_yaw.*joint": 11.455294012539582,
                ".*elbow_pitch.*joint": 11.455294012539582,
                # ".*elbow_yaw.*joint": 9.739,
                # ".*wrist_roll.*joint": 9.739,
                # ".*wrist_pitch.*joint": 9.739,
            },
            stiffness={
                ".*shoulder_pitch.*joint": 150,
                ".*shoulder_roll.*joint": 150,
                ".*shoulder_yaw.*joint": 130,
                ".*elbow_pitch.*joint": 130,
                # ".*elbow_yaw.*joint": 50,
                # ".*wrist_roll.*joint": 20,
                # ".*wrist_pitch.*joint": 20,
            },  # <-- 你来填
            damping={
                ".*shoulder_pitch.*joint": 7.4,
                ".*shoulder_roll.*joint": 7.4,
                ".*shoulder_yaw.*joint": 5.9,
                ".*elbow_pitch.*joint": 5.9,
                # ".*elbow_yaw.*joint": 5,
                # ".*wrist_roll.*joint": 2,
                # ".*wrist_pitch.*joint": 2,
            },    # <-- 你来填
        ),
    },
)



D3_ACTION_SCALE = {}
for a in DEX_EVT_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            D3_ACTION_SCALE[n] = 0.25  ## 先不使用beyondmimic的pd策略