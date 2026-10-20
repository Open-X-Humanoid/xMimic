import gymnasium as gym

from . import agents, flat_env_cfg

##
# Register Gym environments for Tiangong3.
##

gym.register(
    id="Tracking-Flat-Tiangong3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Tiangong3FlatEnvConfig,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Tiangong3FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-Tiangong3-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Tiangong3FlatWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Tiangong3FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-Tiangong3-Low-Freq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Tiangong3FlatLowFreqEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Tiangong3FlatLowFreqPPORunnerCfg",
    },
)
