import gymnasium as gym

from . import agents, flat_env_cfg

##
# Register Gym environments for Dex-V3.
##

gym.register(
    id="Tracking-Flat-DexEVT-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.DexEVTFlatEnvConfig,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexEVTFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-DexEVT-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.DexEVTFlatWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexEVTFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-DexEVT-Low-Freq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.DexEVTFlatLowFreqEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexEVTFlatLowFreqPPORunnerCfg",
    },
)
