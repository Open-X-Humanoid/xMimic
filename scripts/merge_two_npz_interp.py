"""Merge motions with smooth transitions, with an optional joint-only mode.

两种模式：
1) 默认：对齐第二段根位姿到第一段末帧，所有序列插值并合并成一段完整轨迹。
2) `--blend_joints_only`：根保持第二段，关节插值后前置，输出仅基于第二段的新轨迹（不再拼接第一段）。

示例：
python scripts/merge_two_npz_interp.py \\
    --first /path/to/a.npz --second /path/to/b.npz \\
    --output /path/to/output.npz --blend_frames 100            # 模式1：完整合并
python scripts/merge_two_npz_interp.py \\
    --first /path/to/a.npz --second /path/to/b.npz \\
    --output /path/to/output.npz --blend_frames 100 --blend_joints_only  # 模式2：只改关节，保留第二段根
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


# ------------------------- 基础工具 ------------------------- #
def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.asarray(q)[..., [1, 2, 3, 0]]


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.asarray(q)[..., [3, 0, 1, 2]]


def angle_diff(target: float, source: float) -> float:
    """最短角度差 target - source，范围 [-pi, pi]."""
    return np.arctan2(np.sin(target - source), np.cos(target - source))


def get_root_pose(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """返回 (root_pos[3], root_quat_wxyz[4])，优先使用 body_*."""
    if "body_pos_w" in data and "body_quat_w" in data:
        return np.asarray(data["body_pos_w"], dtype=np.float64)[:, 0], np.asarray(
            data["body_quat_w"], dtype=np.float64
        )[:, 0]
    if "root_pos" in data and "root_rot" in data:
        return np.asarray(data["root_pos"], dtype=np.float64), np.asarray(data["root_rot"], dtype=np.float64)
    raise KeyError("未找到 root 位姿，需要包含 body_pos_w/body_quat_w 或 root_pos/root_rot")


def get_root_yaw(quat_wxyz: np.ndarray) -> float:
    return R.from_quat(quat_wxyz_to_xyzw(quat_wxyz)).as_euler("ZYX", degrees=False)[0]


def load_motion(path: Path) -> dict[str, Any]:
    """支持 npz 或保存了 dict 的 pkl/npy。"""
    raw = np.load(path, allow_pickle=True)
    if isinstance(raw, np.lib.npyio.NpzFile):
        data = {k: raw[k] for k in raw.files}
        raw.close()
        return data
    if isinstance(raw, np.ndarray):
        obj = raw.item()
        if not isinstance(obj, dict):
            raise TypeError(f"{path} 内对象不是字典")
        return obj
    raise TypeError(f"不支持的文件类型: {type(raw)}")


# ------------------------- 旋转/平移对齐 ------------------------- #
def align_second_to_first(
    second: dict[str, Any], target_pos: np.ndarray, target_yaw: float, root_index: int = 0
) -> dict[str, Any]:
    """旋转+平移第二段，使其根节点起点落在目标位姿 (位置+yaw)。"""
    data = {k: np.array(v, copy=True) for k, v in second.items()}

    # 当前起点
    body_pos = data.get("body_pos_w")
    body_quat = data.get("body_quat_w")
    if body_pos is None or body_quat is None:
        raise KeyError("需要包含 body_pos_w 和 body_quat_w 用于对齐")

    start_pos = body_pos[0, root_index].astype(np.float64)
    start_quat = body_quat[0, root_index].astype(np.float64)

    curr_yaw = get_root_yaw(start_quat)
    delta_yaw = angle_diff(target_yaw, curr_yaw)
    r_delta = R.from_euler("Z", delta_yaw)

    # 旋转位置 & 线速度 & 角速度
    for key in ("body_pos_w", "body_lin_vel_w", "body_ang_vel_w"):
        if key in data:
            arr = np.asarray(data[key], dtype=np.float64)
            flat = arr.reshape(-1, 3)
            arr_rot = r_delta.apply(flat).reshape(arr.shape)
            data[key] = arr_rot

    # 旋转姿态
    if "body_quat_w" in data:
        arr = np.asarray(data["body_quat_w"], dtype=np.float64)
        flat = quat_wxyz_to_xyzw(arr.reshape(-1, 4))
        r_obj = R.from_quat(flat)
        r_new = (r_delta * r_obj).as_quat()
        data["body_quat_w"] = quat_xyzw_to_wxyz(r_new).reshape(arr.shape)

    # 平移，确保根起点位置匹配
    new_root_pos = data["body_pos_w"][0, root_index]
    translation = target_pos - new_root_pos
    data["body_pos_w"] += translation

    return data


# ------------------------- 插值 ------------------------- #
def _quat_transition(start: np.ndarray, end: np.ndarray, frames: int) -> np.ndarray:
    """对任意形状 (...,4) 的四元数做逐元素 SLERP 过渡。"""
    if frames <= 0:
        return np.empty((0,) + start.shape, dtype=np.float64)
    times = np.linspace(0.0, 1.0, frames + 2)[1:-1]
    start_flat = start.reshape(-1, 4)
    end_flat = end.reshape(-1, 4)
    out = np.zeros((frames, start_flat.shape[0], 4), dtype=np.float64)
    for i, (qs, qe) in enumerate(zip(start_flat, end_flat)):
        rots = R.from_quat(quat_wxyz_to_xyzw(np.stack([qs, qe])))
        slerp = Slerp([0.0, 1.0], rots)
        out[:, i, :] = quat_xyzw_to_wxyz(slerp(times).as_quat())
    return out.reshape((frames,) + start.shape)


def _linear_transition(start: np.ndarray, end: np.ndarray, frames: int) -> np.ndarray:
    if frames <= 0:
        return np.empty((0,) + start.shape, dtype=np.float64)
    seg = np.linspace(start, end, frames + 2, axis=0)[1:-1]
    return seg


def blend_series(a: np.ndarray, b: np.ndarray, frames: int, is_quat: bool) -> np.ndarray:
    """生成过渡片段，连接 a[-1] -> b[0]。"""
    start, end = a[-1], b[0]
    if is_quat:
        return _quat_transition(start, end, frames)
    return _linear_transition(start, end, frames)


def _infer_seq_len(data: dict[str, Any]) -> tuple[int, str]:
    priority = ("joint_pos", "root_pos", "body_pos_w", "body_quat_w", "dof_pos", "dof_vel", "body_lin_vel_w")
    for key in priority:
        if key in data:
            arr = np.asarray(data[key])
            if arr.ndim >= 1:
                return int(arr.shape[0]), key
    for key, val in data.items():
        arr = np.asarray(val)
        if arr.ndim >= 1 and arr.shape[0] > 1:
            return int(arr.shape[0]), key
    raise ValueError("无法推断时间序列长度，请确认包含 joint_pos/root_pos/body_pos_w 等键")


# ------------------------- 模式1：完整合并（根对齐 + 全量拼接） ------------------------- #
def merge_with_transition(
    first: dict[str, Any],
    second: dict[str, Any],
    blend_frames: int,
    root_index: int = 0,
) -> dict[str, Any]:
    first_len, first_key = _infer_seq_len(first)
    second_len, second_key = _infer_seq_len(second)
    print(f"[INFO] 第一段时间长度 {first_len} (键: {first_key}), 第二段 {second_len} (键: {second_key})")

    root_pos_a, root_quat_a = get_root_pose(first)
    target_pos = root_pos_a[-1]
    target_yaw = get_root_yaw(root_quat_a[-1])

    second_aligned = align_second_to_first(second, target_pos, target_yaw, root_index=root_index)

    merged: dict[str, Any] = {}
    for key, arr_a in first.items():
        arr_b = second_aligned.get(key)
        if isinstance(arr_a, np.ndarray) and isinstance(arr_b, np.ndarray):
            if arr_a.ndim >= 1 and arr_b.ndim >= 1 and arr_a.shape[0] == first_len and arr_b.shape[0] == second_len:
                is_quat = ("quat" in key) and arr_a.shape[-1] == 4
                transition = blend_series(arr_a.astype(np.float64), arr_b.astype(np.float64), blend_frames, is_quat)
                tail = arr_b[1:] if arr_b.shape[0] > 1 else arr_b[:0]
                merged_arr = np.concatenate([arr_a, transition, tail], axis=0)
                merged[key] = merged_arr.astype(arr_a.dtype, copy=False)
            else:
                merged[key] = arr_a
        else:
            merged[key] = arr_a

    return merged


# ------------------------- 模式2：仅关节插值，根保持第二段 ------------------------- #
def merge_joint_only(
    first: dict[str, Any],
    second: dict[str, Any],
    blend_frames: int,
    joint_key_patterns: tuple[str, ...] = ("joint_pos", "joint_vel", "dof_pos", "dof_vel"),
) -> dict[str, Any]:
    first_len, first_key = _infer_seq_len(first)
    second_len, second_key = _infer_seq_len(second)
    print(f"[INFO] 第一段时间长度 {first_len} (键: {first_key}), 第二段 {second_len} (键: {second_key})")

    blended: dict[str, Any] = {}
    for key, arr_b in second.items():
        if isinstance(arr_b, np.ndarray) and arr_b.ndim >= 1 and arr_b.shape[0] == second_len:
            is_joint = any(pat in key for pat in joint_key_patterns)
            if is_joint:
                arr_a = first.get(key)
                if not isinstance(arr_a, np.ndarray) or arr_a.shape[0] != first_len:
                    raise ValueError(f"关节序列 '{key}' 在第一段缺失或长度不匹配，无法插值。")
                is_quat = ("quat" in key) and arr_a.shape[-1] == 4
                transition = blend_series(arr_a.astype(np.float64), arr_b.astype(np.float64), blend_frames, is_quat)
                new_arr = np.concatenate([transition, arr_b], axis=0)
                blended[key] = new_arr.astype(arr_b.dtype, copy=False)
            else:
                if blend_frames > 0:
                    pad = np.repeat(arr_b[0:1], blend_frames, axis=0)
                    blended[key] = np.concatenate([pad, arr_b], axis=0).astype(arr_b.dtype, copy=False)
                else:
                    blended[key] = arr_b
        else:
            blended[key] = arr_b

    return blended


# ------------------------- CLI ------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="合并两个 npz/pkl 动作文件并添加平滑过渡。")
    parser.add_argument("--first", type=Path, required=True, help="第一段文件路径")
    parser.add_argument("--second", type=Path, required=True, help="第二段文件路径")
    parser.add_argument("--output", type=Path, required=True, help="输出 npz 文件路径")
    parser.add_argument("--blend_frames", type=int, default=100, help="过渡帧数，默认 100")
    parser.add_argument(
        "--root_index", type=int, default=0, help="根节点在 body 数组中的索引，默认 0（通常是 pelvis/base）"
    )
    parser.add_argument(
        "--blend_joints_only",
        action="store_true",
        help="仅对关节做过渡并前置到第二段，根保持第二段且不合并第一段主体。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    first = load_motion(args.first)
    second = load_motion(args.second)

    if args.blend_joints_only:
        merged = merge_joint_only(first, second, blend_frames=args.blend_frames)
    else:
        merged = merge_with_transition(first, second, blend_frames=args.blend_frames, root_index=args.root_index)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **merged)
    print(f"[INFO] 已保存合并结果: {args.output}")
    for k, v in merged.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: {v.shape} {v.dtype}")
        else:
            print(f"  {k}: {type(v)}")


if __name__ == "__main__":
    main()
