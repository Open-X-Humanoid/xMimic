import os

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        policy_path = os.path.dirname(path)
        filename = os.path.basename(policy_path.rstrip(os.sep)) + ".onnx"
        export_policy_as_onnx(
            self.alg.policy,
            normalizer=getattr(self.alg.policy, "actor_obs_normalizer", None),
            path=policy_path,
            filename=filename,
        )
        run_name = os.path.basename(os.path.normpath(self.log_dir)) if getattr(self, "log_dir", None) else "offline_run"
        attach_onnx_metadata(self.env.unwrapped, run_name, path=policy_path, filename=filename)


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        super().__init__(env, train_cfg, log_dir, device)

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        policy_path = os.path.dirname(path)
        filename = os.path.basename(policy_path.rstrip(os.sep)) + ".onnx"
        export_motion_policy_as_onnx(
            self.env.unwrapped,
            self.alg.policy,
            normalizer=getattr(self.alg.policy, "actor_obs_normalizer", None),
            path=policy_path,
            filename=filename,
        )
        run_name = os.path.basename(os.path.normpath(self.log_dir)) if getattr(self, "log_dir", None) else "offline_run"
        attach_onnx_metadata(self.env.unwrapped, run_name, path=policy_path, filename=filename)
