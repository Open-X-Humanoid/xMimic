"""Load motion arrays in the requested body and joint name order."""

from collections.abc import Sequence
import warnings

import numpy as np
import torch


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        body_names: Sequence[str],
        joint_names: Sequence[str],
        device: str = "cpu",
        *,
        body_indexes: Sequence[int] | None = None,
    ):
        with np.load(motion_file, allow_pickle=False) as data:
            self.fps = data["fps"].copy()
            joint_shape = data["joint_pos"].shape
            body_shape = data["body_pos_w"].shape
            if len(joint_shape) != 2 or len(body_shape) != 3:
                raise ValueError(f"Invalid joint_pos/body_pos_w shapes in {motion_file}")
            self.time_step_total = joint_shape[0]
            if self.time_step_total == 0:
                raise ValueError(f"Motion has no frames: {motion_file}")
            missing_metadata = [key for key in ("body_names", "joint_names") if key not in data]
            if "joint_names" in data:
                joint_indices = self._name_indices(data, "joint_names", joint_names, joint_shape[1])
            else:
                if joint_shape[1] != len(joint_names):
                    raise ValueError("Motion joint count does not match the robot; cannot use index-based loading")
                joint_indices = list(range(joint_shape[1]))
            if "body_names" in data:
                body_indices = self._name_indices(data, "body_names", body_names, body_shape[1])
            else:
                if body_indexes is None:
                    raise ValueError("Motion has no body_names; provide the robot's body_indexes for legacy loading")
                body_indices = list(body_indexes)
                if len(body_indices) != len(body_names) or any(
                    not isinstance(index, (int, np.integer)) or not 0 <= index < body_shape[1]
                    for index in body_indices
                ):
                    raise ValueError("Robot body_indexes do not match the requested bodies or exceed the motion body count")
            if missing_metadata:
                warnings.warn(
                    f"{motion_file}: missing {', '.join(missing_metadata)}; falling back to the previous "
                    "index-based loading for those fields. NPZ body/joint order must match the robot model; "
                    "name alignment cannot be verified. Regenerate the NPZ to save names.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            def load_array(key, indices, width, components=None):
                array = data[key]
                expected = (self.time_step_total, width)
                if components is not None:
                    expected += (components,)
                if array.shape != expected:
                    raise ValueError(f"{key} has shape {array.shape}; expected {expected}")
                return torch.tensor(array[:, indices], dtype=torch.float32, device=device)

            self.joint_pos = load_array("joint_pos", joint_indices, joint_shape[1])
            self.joint_vel = load_array("joint_vel", joint_indices, joint_shape[1])
            self.body_pos_w = load_array("body_pos_w", body_indices, body_shape[1], 3)
            self.body_quat_w = load_array("body_quat_w", body_indices, body_shape[1], 4)
            self.body_lin_vel_w = load_array("body_lin_vel_w", body_indices, body_shape[1], 3)
            self.body_ang_vel_w = load_array("body_ang_vel_w", body_indices, body_shape[1], 3)
        self.body_names = list(body_names)
        self.joint_names = list(joint_names)

    @staticmethod
    def _name_indices(data, key: str, requested: Sequence[str], width: int) -> list[int]:
        if key not in data:
            raise ValueError(f"Motion NPZ is missing '{key}'. Regenerate it with the updated conversion script.")
        names = data[key]
        if names.ndim != 1 or len(names) != width or names.dtype.kind not in ("U", "S"):
            raise ValueError(f"'{key}' must contain {width} string names matching the data columns")
        names = names.astype(str).tolist()
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError(f"'{key}' contains empty or duplicate names")
        indices = {name: index for index, name in enumerate(names)}
        missing = [name for name in requested if name not in indices]
        if missing:
            raise ValueError(f"Motion '{key}' is missing requested names: {missing}")
        return [indices[name] for name in requested]
