#!/usr/bin/env python3
import os
import argparse
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# ===================== 配置 =====================
input_folder = "motion_data/dex_evt_motion_data/1"
output_folder = "dex_evt_motion_data/1/merge"
output_file = os.path.join(output_folder, "merge_climb.npz")
# 通过CLI开启全三轴根姿态对齐（默认只对齐yaw）
parser = argparse.ArgumentParser(description="Merge motion npz files with optional full root alignment.")
parser.add_argument(
    "--full_root_align",
    action="store_true",
    help="Align full root orientation (xyz) instead of yaw-only when merging.",
)
parser.add_argument(
    "--align_yaw_xy_only",
    action="store_true",
    help="仅对齐根节点 yaw 并平移 xy，z 不动，避免高度被改动（优先于 full_root_align）。",
)
args = parser.parse_args()
ALIGN_YAW_XY_ONLY = args.align_yaw_xy_only
FULL_ROOT_ALIGN = args.full_root_align
# ================================================

def load_npz(path):
    data = np.load(path, allow_pickle=True)
    out = {}
    for k in data.files:
        v = data[k]
        # 标量/bytes 转成 ndarray 便于后续处理；否则保持 ndarray
        if not isinstance(v, np.ndarray):
            v = np.array(v)
        out[k] = v
    return out

def quat_wxyz_to_xyzw(q):
    """Convert quaternion from [w, x, y, z] to [x, y, z, w] for SciPy."""
    return np.asarray(q)[..., [1, 2, 3, 0]]

def quat_xyzw_to_wxyz(q):
    """Convert quaternion from [x, y, z, w] to [w, x, y, z]."""
    return np.asarray(q)[..., [3, 0, 1, 2]]

def angle_diff(target, source):
    """Shortest angular difference target - source, wrap to [-pi, pi]."""
    return np.arctan2(np.sin(target - source), np.cos(target - source))

def get_root_yaw(quat):
    return R.from_quat(quat_wxyz_to_xyzw(quat)).as_euler("ZYX", degrees=False)[0]

npz_files = sorted([str(p) for p in Path(input_folder).rglob("*.npz") if p.is_file()])
# 避免把上一次的输出再次当作输入
npz_files = [p for p in npz_files if os.path.abspath(p) != os.path.abspath(output_file)]
print("Found", len(npz_files), "npz:")
for f in npz_files:
    print(" ", Path(f).relative_to(input_folder))

# 若目录下没有 npz，直接退出，避免后续 merged 仍为 None
if not npz_files:
    raise SystemExit(f"No .npz files found in {input_folder}")

merged = None
prev_end_pos = None
prev_end_yaw = None
prev_end_quat_wxyz = None

for idx, path in enumerate(npz_files):
    print(f"\n--- [{idx+1}/{len(npz_files)}] Loading:", path)
    cur = load_npz(path)

    body_pos = cur["body_pos_w"].astype(np.float64)
    body_quat_wxyz = cur["body_quat_w"].astype(np.float64)
    body_quat = quat_wxyz_to_xyzw(body_quat_wxyz)

    if merged is None:
        print("  → Initialize merged buffer")
        merged = {k: np.array(v, copy=True) for k, v in cur.items()}

        prev_end_pos = merged["body_pos_w"][-1, 0].copy()
        prev_end_yaw = get_root_yaw(merged["body_quat_w"][-1, 0])
        prev_end_quat_wxyz = merged["body_quat_w"][-1, 0].copy()
        continue

    # ---------- 1. 姿态对齐 ----------
    if ALIGN_YAW_XY_ONLY:
        curr_yaw = get_root_yaw(body_quat_wxyz[0, 0])
        delta_yaw = angle_diff(prev_end_yaw, curr_yaw)
        print(f"  yaw-only delta = {np.degrees(delta_yaw):.3f} deg (z保持不变)")
        R_delta = R.from_euler("Z", delta_yaw, degrees=False)
    elif FULL_ROOT_ALIGN:
        curr_quat_xyzw = body_quat[0, 0]  # [x,y,z,w]
        prev_quat_xyzw = quat_wxyz_to_xyzw(prev_end_quat_wxyz)
        R_delta = R.from_quat(prev_quat_xyzw) * R.from_quat(curr_quat_xyzw).inv()
        delta_euler = R_delta.as_euler("xyz", degrees=True)
        print(f"  delta rot xyz (deg): {delta_euler}")
    else:
        curr_yaw = get_root_yaw(body_quat_wxyz[0, 0])
        delta_yaw = angle_diff(prev_end_yaw, curr_yaw)
        print(f"  yaw delta = {np.degrees(delta_yaw):.3f} deg")
        R_delta = R.from_euler("Z", delta_yaw, degrees=False)

    # ---------- 2. 世界坐标旋转 ----------
    T, B = body_pos.shape[:2]
    body_pos_flat = body_pos.reshape(-1, 3)
    body_pos_rot = R_delta.apply(body_pos_flat).reshape(T, B, 3)

    # 同步旋转线速度 / 角速度（世界系量）保持一致
    body_lin_vel = cur["body_lin_vel_w"].astype(np.float64)
    body_lin_vel_rot = R_delta.apply(body_lin_vel.reshape(-1, 3)).reshape(T, B, 3)
    cur["body_lin_vel_w"] = body_lin_vel_rot.astype(np.float32)

    body_ang_vel = cur.get("body_ang_vel_w")
    if body_ang_vel is not None:
        body_ang_vel = body_ang_vel.astype(np.float64)
        body_ang_vel_rot = R_delta.apply(body_ang_vel.reshape(-1, 3)).reshape(T, B, 3)
        cur["body_ang_vel_w"] = body_ang_vel_rot.astype(np.float32)

    # ---------- 3. 平移对齐 ----------
    rot_root_start = body_pos_rot[0, 0]
    translation = prev_end_pos - rot_root_start
    if FULL_ROOT_ALIGN or ALIGN_YAW_XY_ONLY:
        translation[2] = 0.0  # z 不对齐，仅对齐 xy
    body_pos_rot += translation
    cur["body_pos_w"] = body_pos_rot.astype(np.float32)

    # ---------- 4. 旋转 world quat ----------
    # 世界坐标整体旋转 R_delta：姿态矩阵应左乘 R_delta
    flat_quat = body_quat.reshape(-1, 4)
    r_obj = R.from_quat(flat_quat)
    r_new = (R_delta * r_obj).as_quat().reshape(T, B, 4)
    cur["body_quat_w"] = quat_xyzw_to_wxyz(r_new).astype(np.float32)

    # ---------- 5. 更新 prev_end ----------
    prev_end_pos = cur["body_pos_w"][-1, 0].copy()
    prev_end_yaw = get_root_yaw(cur["body_quat_w"][-1, 0])
    prev_end_quat_wxyz = cur["body_quat_w"][-1, 0].copy()

    # ---------- 6. 拼接 ----------
    for k in merged.keys():
        if isinstance(merged[k], np.ndarray) and isinstance(cur.get(k), np.ndarray):
            merged[k] = np.concatenate([merged[k], cur[k]], axis=0)
        # 非 ndarray 的直接忽略拼接，保持初始值

np.savez(output_file, **merged)
print("\n✅ 合并完成 →", output_file)
for k, v in merged.items():
    print(" ", k, v.shape, v.dtype)
