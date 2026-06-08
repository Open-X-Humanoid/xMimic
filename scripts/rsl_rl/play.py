"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--play_full_motion",
    action="store_true",
    default=False,
    help="Start the reference motion at phase 0 and stop playback after one full trajectory.",
)
parser.add_argument(
    "--play_env_id",
    type=int,
    default=0,
    help="Environment index to monitor for --play_full_motion stopping condition.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument(
    "--keep_running",
    action="store_true",
    default=True,
    help="Prevent automatic exit after video capture or one full motion playback.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import pathlib
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx

# Reward terms that were recently introduced and should be logged during playbacks.
_NEW_REWARD_TERMS = ("joint_torque_l2", "joint_vel_limit", "joint_torque_limit")


class ClipToLimit(gym.ActionWrapper):
    """Clip raw actions to a fixed scalar limit before env processing."""

    def __init__(self, env, limit: float):
        super().__init__(env)
        self.limit = float(limit)

    def action(self, action):
        if isinstance(action, torch.Tensor):
            return torch.clamp(action, -self.limit, self.limit)
        return np.clip(action, -self.limit, self.limit)


def _prepare_full_motion_play(vec_env: RslRlVecEnvWrapper):
    """Align the motion command with its first frame for deterministic playback."""
    base_env = getattr(vec_env, "unwrapped", vec_env)
    command_manager = getattr(base_env, "command_manager", None)
    if command_manager is None:
        return None, None
    try:
        motion_term = command_manager.get_term("motion")
    except KeyError:
        return None, None

    env_ids = torch.arange(motion_term.num_envs, device=motion_term.device, dtype=torch.long)
    motion_term.time_steps.zero_()
    horizon_s = float(motion_term.motion.time_step_total) * base_env.step_dt
    motion_term.time_left[env_ids] = horizon_s

    if horizon_s > base_env.cfg.episode_length_s:
        base_env.cfg.episode_length_s = horizon_s
        if hasattr(base_env, "episode_length_buf"):
            base_env.episode_length_buf.zero_()

    joint_pos = motion_term.joint_pos.clone()
    joint_vel = motion_term.joint_vel.clone()
    motion_term.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)

    root_state = torch.cat(
        [
            motion_term.body_pos_w[:, 0],
            motion_term.body_quat_w[:, 0],
            motion_term.body_lin_vel_w[:, 0],
            motion_term.body_ang_vel_w[:, 0],
        ],
        dim=-1,
    )
    motion_term.robot.write_root_state_to_sim(root_state[env_ids], env_ids=env_ids)
    return motion_term, int(motion_term.motion.time_step_total)


def _log_new_reward_terms(vec_env: RslRlVecEnvWrapper, env_idx: int = 0):
    """Prints the contribution of the newly added reward terms for a representative environment."""
    reward_manager = getattr(vec_env.unwrapped, "reward_manager", None)
    if reward_manager is None:
        return

    log_values = []
    for name, values in reward_manager.get_active_iterable_terms(env_idx=env_idx):
        if name in _NEW_REWARD_TERMS and len(values) > 0:
            log_values.append(f"{name}: {values[0]:.4f}")

    # if log_values:
    #     print(f"[REWARD] env {env_idx} | " + ", ".join(log_values))


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    if args_cli.motion_file is not None:
        print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
        env_cfg.commands.motion.motion_file = args_cli.motion_file

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    log_dir = os.path.dirname(resume_path)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # clip actions to keep inference consistent with training limits from PPO cfg
    env = ClipToLimit(env, limit=getattr(agent_cfg, "clip_action", np.inf))

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    motion_term = None
    motion_max_steps = None
    prev_motion_step = None
    play_env_id = args_cli.play_env_id
    if args_cli.play_full_motion:
        motion_term, motion_max_steps = _prepare_full_motion_play(env)
        if motion_term is not None:
            play_env_id = max(0, min(play_env_id, motion_term.num_envs - 1))
            prev_motion_step = motion_term.time_steps[play_env_id].item()

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    # resolve observation normalizer from actor-critic (rsl_rl >= 2.0)
    actor_obs_normalizer = getattr(ppo_runner.alg.policy, "actor_obs_normalizer", None)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    export_motion_policy_as_onnx(
        env.unwrapped,
        ppo_runner.alg.policy,
        normalizer=actor_obs_normalizer,
        path=export_model_dir,
        filename="policy.onnx",
    )
    attach_onnx_metadata(env.unwrapped, "local", export_model_dir)
    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        _log_new_reward_terms(env)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length and not args_cli.keep_running:
                break
        if args_cli.play_full_motion and motion_term is not None:
            current_step = motion_term.time_steps[play_env_id].item()
            if current_step < prev_motion_step and not args_cli.keep_running:
                break
            prev_motion_step = current_step
            if motion_max_steps is not None and current_step >= motion_max_steps - 1 and not args_cli.keep_running:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
