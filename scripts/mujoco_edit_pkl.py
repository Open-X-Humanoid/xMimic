#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import pickle
import sys
import time
import tempfile
import shutil
import queue
from pathlib import Path
from typing import Callable

import numpy as np


DEFAULT_URDF = (
    "source/whole_body_tracking/whole_body_tracking/"
    "assets/tiangong2dex_urdf/urdf/tiangong2dex.urdf"
)

JOINT_NAMES_TG = [
    "hip_pitch_l_joint",
    "hip_roll_l_joint",
    "hip_yaw_l_joint",
    "knee_pitch_l_joint",
    "ankle_pitch_l_joint",
    "ankle_roll_l_joint",
    "hip_pitch_r_joint",
    "hip_roll_r_joint",
    "hip_yaw_r_joint",
    "knee_pitch_r_joint",
    "ankle_pitch_r_joint",
    "ankle_roll_r_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "shoulder_pitch_l_joint",
    "shoulder_roll_l_joint",
    "shoulder_yaw_l_joint",
    "elbow_pitch_l_joint",
    "elbow_yaw_l_joint",
    "wrist_pitch_l_joint",
    "wrist_roll_l_joint",
    "shoulder_pitch_r_joint",
    "shoulder_roll_r_joint",
    "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint",
    "elbow_yaw_r_joint",
    "wrist_pitch_r_joint",
    "wrist_roll_r_joint",
]

JOINT_NAMES_DEX_EVT = [
    "hip_pitch_l_joint",
    "hip_roll_l_joint",
    "hip_yaw_l_joint",
    "knee_pitch_l_joint",
    "ankle_pitch_l_joint",
    "ankle_roll_l_joint",
    "hip_pitch_r_joint",
    "hip_roll_r_joint",
    "hip_yaw_r_joint",
    "knee_pitch_r_joint",
    "ankle_pitch_r_joint",
    "ankle_roll_r_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "shoulder_pitch_l_joint",
    "shoulder_roll_l_joint",
    "shoulder_yaw_l_joint",
    "elbow_pitch_l_joint",
    "shoulder_pitch_r_joint",
    "shoulder_roll_r_joint",
    "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint",
]

ROBOT_JOINTS = {
    "tgdex": JOINT_NAMES_TG,
    "dex_evt": JOINT_NAMES_DEX_EVT,
}


def _import_or_exit() -> tuple[object, object, object, object]:
    try:
        import mujoco  # type: ignore
        import mujoco.viewer  # type: ignore
    except Exception as exc:  # noqa: BLE001
        print("ERROR: mujoco python package not available.", file=sys.stderr)
        print("       Install with: pip install mujoco", file=sys.stderr)
        raise SystemExit(1) from exc
    try:
        import glfw  # type: ignore
    except Exception as exc:  # noqa: BLE001
        print("ERROR: glfw python package not available.", file=sys.stderr)
        print("       Install with: pip install glfw", file=sys.stderr)
        raise SystemExit(1) from exc
    try:
        import tkinter as tk  # type: ignore
        from tkinter import messagebox  # type: ignore
    except Exception as exc:  # noqa: BLE001
        print("ERROR: tkinter not available on this system.", file=sys.stderr)
        raise SystemExit(1) from exc
    return mujoco, glfw, tk, messagebox


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def quat_wxyz_to_euler_xyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = max(-1.0, min(1.0, t2))
    pitch = math.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return np.array([roll, pitch, yaw], dtype=np.float64)


def euler_xyz_to_quat_wxyz(euler: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = euler
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z], dtype=np.float64)


def _ensure_numpy_core_alias() -> None:
    # Map numpy._core to numpy.core for older NumPy installs loading newer pickles.
    import sys

    try:
        import numpy.core as np_core
    except Exception:  # noqa: BLE001
        return
    sys.modules.setdefault("numpy._core", np_core)
    if hasattr(np_core, "multiarray"):
        sys.modules.setdefault("numpy._core.multiarray", np_core.multiarray)
    if hasattr(np_core, "_multiarray_umath"):
        sys.modules.setdefault("numpy._core._multiarray_umath", np_core._multiarray_umath)


def _rewrite_urdf_mesh_paths(urdf_path: Path) -> Path:
    text = urdf_path.read_text(encoding="utf-8")
    base_dir = urdf_path.parent
    package_root = urdf_path.parent.parent
    tmp_dir = Path(tempfile.mkdtemp(prefix="mujoco_urdf_"))
    name_to_src: dict[str, Path] = {}
    src_to_name: dict[Path, str] = {}

    def replace_mesh(match: "re.Match[str]") -> str:
        raw_path = match.group(1)
        new_path = raw_path
        abs_path: Path | None = None
        if raw_path.startswith("package://"):
            parts = raw_path[len("package://") :].split("/", 1)
            if len(parts) == 2:
                pkg, rel = parts
                if pkg == package_root.name:
                    abs_path = (package_root / rel).resolve()
        elif not raw_path.startswith(("/", "\\")):
            abs_path = (base_dir / raw_path).resolve()
        else:
            abs_path = Path(raw_path).resolve()

        if abs_path is not None and abs_path.exists():
            if abs_path in src_to_name:
                new_path = src_to_name[abs_path]
            else:
                name = abs_path.name
                if name in name_to_src and name_to_src[name] != abs_path:
                    stem, suffix = abs_path.stem, abs_path.suffix
                    counter = 1
                    candidate = f"{stem}_{counter}{suffix}"
                    while candidate in name_to_src and name_to_src[candidate] != abs_path:
                        counter += 1
                        candidate = f"{stem}_{counter}{suffix}"
                    name = candidate
                name_to_src[name] = abs_path
                src_to_name[abs_path] = name
                shutil.copy2(abs_path, tmp_dir / name)
            new_path = src_to_name[abs_path]
        return f'mesh filename="{new_path}"'

    import re

    updated = re.sub(r'mesh\s+filename=[\'"]([^\'"]+)[\'"]', replace_mesh, text)
    if updated == text:
        return urdf_path

    out_path = tmp_dir / urdf_path.name
    out_path.write_text(updated, encoding="utf-8")
    return out_path


def _rewrite_mjcf_meshdir(xml_path: Path, meshdir: Path | None) -> Path:
    text = xml_path.read_text(encoding="utf-8")
    import re

    compiler_match = re.search(r"<compiler\\b[^>]*>", text)
    new_meshdir: str | None = None
    raw_meshdir: str | None = None
    if compiler_match:
        tag = compiler_match.group(0)
        md = re.search(r"meshdir=[\\\"\\']([^\\\"\\']+)[\\\"\\']", tag)
        if md:
            raw_meshdir = md.group(1)

    if meshdir is not None:
        new_meshdir = str(meshdir.resolve())
    elif raw_meshdir is not None:
        if raw_meshdir.startswith(("/", "\\")):
            new_meshdir = raw_meshdir
        else:
            new_meshdir = str((xml_path.parent / raw_meshdir).resolve())

    if new_meshdir is None:
        return xml_path

    def replace_compiler(match: "re.Match[str]") -> str:
        tag = match.group(0)
        if "meshdir=" in tag:
            return re.sub(r"meshdir=[\\\"\\'][^\\\"\\']+[\\\"\\']", f'meshdir="{new_meshdir}"', tag)
        if tag.endswith("/>"):
            return tag[:-2] + f' meshdir="{new_meshdir}"/>'
        return tag[:-1] + f' meshdir="{new_meshdir}">'

    if compiler_match:
        updated = re.sub(r"<compiler\\b[^>]*>", replace_compiler, text, count=1)
    else:
        mj_match = re.search(r"<mujoco\\b[^>]*>", text)
        insert = f'  <compiler meshdir="{new_meshdir}"/>\\n'
        if mj_match:
            idx = mj_match.end()
            updated = text[:idx] + "\\n" + insert + text[idx:]
        else:
            updated = insert + text

    if updated == text:
        return xml_path

    tmp_dir = Path(tempfile.mkdtemp(prefix="mujoco_xml_"))
    out_path = tmp_dir / xml_path.name
    out_path.write_text(updated, encoding="utf-8")
    return out_path


def apply_interpolation(
    arr: np.ndarray, start: int, key: int, end: int, target: np.ndarray, mode: str
) -> None:
    if arr.ndim != 2:
        raise ValueError("apply_interpolation expects a 2D array")
    if mode == "overwrite":
        start_val = arr[start].copy()
        end_val = arr[end].copy()
        if key > start:
            t = np.linspace(0.0, 1.0, key - start + 1, dtype=arr.dtype)[:, None]
            arr[start : key + 1] = start_val * (1.0 - t) + target * t
        else:
            arr[key] = target
        if end > key:
            t = np.linspace(0.0, 1.0, end - key + 1, dtype=arr.dtype)[:, None]
            arr[key : end + 1] = target * (1.0 - t) + end_val * t
    elif mode == "add":
        delta = target - arr[key]
        if end < start:
            return
        weights = np.zeros((end - start + 1, 1), dtype=arr.dtype)
        if key > start:
            weights[: key - start + 1, 0] = np.linspace(0.0, 1.0, key - start + 1, dtype=arr.dtype)
        else:
            weights[0, 0] = 1.0
        if end > key:
            weights[key - start :, 0] = np.linspace(1.0, 0.0, end - key + 1, dtype=arr.dtype)
        arr[start : end + 1] = arr[start : end + 1] + delta * weights
    else:
        raise ValueError(f"Unknown mode: {mode}")


class EditorWindow:
    def __init__(
        self,
        tk_mod: object,
        messagebox_mod: object,
        joint_names: list[str],
        apply_cb: Callable[[np.ndarray, np.ndarray, np.ndarray, int, int, str, int, bool], None],
        revert_cb: Callable[[int, int], None],
        save_cb: Callable[[Path], None],
        font_size: int,
    ) -> None:
        self.tk = tk_mod
        self.messagebox = messagebox_mod
        self.apply_cb = apply_cb
        self.revert_cb = revert_cb
        self.save_cb = save_cb
        self.joint_names = joint_names
        self.font_size = font_size

        self.root = self.tk.Tk()
        self.root.title("PKL Motion Editor")
        self.root.geometry("520x700")
        self.root.protocol("WM_DELETE_WINDOW", self.hide)
        self.root.withdraw()
        self.visible = False
        self._apply_font_size()

        self.frame_var = self.tk.StringVar(value="0")
        self.start_var = self.tk.StringVar(value="0")
        self.end_var = self.tk.StringVar(value="0")
        self.insert_var = self.tk.StringVar(value="0")
        self.mode_var = self.tk.StringVar(value="overwrite")
        self.output_var = self.tk.StringVar(value="")
        self.apply_root_var = self.tk.BooleanVar(value=True)

        self.root_pos_vars = [self.tk.StringVar() for _ in range(3)]
        self.root_rpy_vars = [self.tk.StringVar() for _ in range(3)]
        self.joint_vars = [self.tk.StringVar() for _ in joint_names]

        self._build_layout()

    def _apply_font_size(self) -> None:
        try:
            import tkinter.font as tkfont
        except Exception:  # noqa: BLE001
            return
        for name in (
            "TkDefaultFont",
            "TkTextFont",
            "TkFixedFont",
            "TkMenuFont",
            "TkHeadingFont",
            "TkCaptionFont",
            "TkSmallCaptionFont",
            "TkIconFont",
            "TkTooltipFont",
        ):
            try:
                font = tkfont.nametofont(name)
                font.configure(size=self.font_size)
            except Exception:  # noqa: BLE001
                continue

    def _build_layout(self) -> None:
        header = self.tk.Frame(self.root)
        self.tk.Label(header, text="Frame:").pack(side="left", padx=5)
        self.tk.Label(header, textvariable=self.frame_var).pack(side="left")
        header.pack(fill="x", pady=6)

        root_frame = self.tk.LabelFrame(self.root, text="Root Pose (m, rad)")
        self._grid_labeled_entries(root_frame, ["x", "y", "z"], self.root_pos_vars, row=0)
        self._grid_labeled_entries(root_frame, ["roll", "pitch", "yaw"], self.root_rpy_vars, row=1)
        root_frame.pack(fill="x", padx=8, pady=6)

        interp_frame = self.tk.LabelFrame(self.root, text="Interpolation")
        self._grid_labeled_entries(
            interp_frame,
            ["start", "end", "insert"],
            [self.start_var, self.end_var, self.insert_var],
            row=0,
        )
        mode_row = self.tk.Frame(interp_frame)
        self.tk.Radiobutton(mode_row, text="overwrite", variable=self.mode_var, value="overwrite").pack(
            side="left", padx=5
        )
        self.tk.Radiobutton(mode_row, text="add", variable=self.mode_var, value="add").pack(side="left", padx=5)
        self.tk.Checkbutton(
            mode_row,
            text="apply root/base",
            variable=self.apply_root_var,
        ).pack(side="left", padx=8)
        mode_row.grid(row=1, column=0, columnspan=6, pady=4)
        interp_frame.pack(fill="x", padx=8, pady=6)

        output_frame = self.tk.LabelFrame(self.root, text="Output")
        self.tk.Entry(output_frame, textvariable=self.output_var, width=64).pack(side="left", padx=6, pady=4)
        output_frame.pack(fill="x", padx=8, pady=6)

        joints_frame = self.tk.LabelFrame(self.root, text="Joint Positions (rad)")
        canvas = self.tk.Canvas(joints_frame, height=320)
        scrollbar = self.tk.Scrollbar(joints_frame, orient="vertical", command=canvas.yview)
        scroll = self.tk.Frame(canvas)

        scroll.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        for i, name in enumerate(self.joint_names):
            row = self.tk.Frame(scroll)
            self.tk.Label(row, text=name, width=24, anchor="w").pack(side="left", padx=4)
            self.tk.Entry(row, textvariable=self.joint_vars[i], width=16).pack(side="left", padx=4)
            row.pack(fill="x", pady=1)

        joints_frame.pack(fill="both", expand=True, padx=8, pady=6)

        buttons = self.tk.Frame(self.root)
        self.tk.Button(buttons, text="Apply", command=self._on_apply).pack(side="left", padx=6)
        self.tk.Button(buttons, text="Revert Range", command=self._on_revert).pack(side="left", padx=6)
        self.tk.Button(buttons, text="Save", command=self._on_save).pack(side="left", padx=6)
        self.tk.Button(buttons, text="Close", command=self.hide).pack(side="left", padx=6)
        buttons.pack(pady=8)

    def _grid_labeled_entries(
        self, parent: object, labels: list[str], vars_list: list[object], row: int
    ) -> None:
        for col, label in enumerate(labels):
            self.tk.Label(parent, text=label).grid(row=row, column=col * 2, padx=4, pady=2, sticky="w")
            self.tk.Entry(parent, textvariable=vars_list[col], width=12).grid(
                row=row, column=col * 2 + 1, padx=4, pady=2
            )

    def open(
        self,
        frame_idx: int,
        root_pos: np.ndarray,
        root_rpy: np.ndarray,
        joint_pos: np.ndarray,
        start: int,
        end: int,
        output_path: Path,
    ) -> None:
        self.frame_var.set(str(frame_idx))
        for i in range(3):
            self.root_pos_vars[i].set(f"{root_pos[i]:.6f}")
            self.root_rpy_vars[i].set(f"{root_rpy[i]:.6f}")
        for i in range(len(self.joint_vars)):
            self.joint_vars[i].set(f"{joint_pos[i]:.6f}")
        self.start_var.set(str(start))
        self.end_var.set(str(end))
        self.output_var.set(str(output_path))
        self.root.deiconify()
        self.visible = True

    def hide(self) -> None:
        self.root.withdraw()
        self.visible = False

    def update(self) -> None:
        if self.visible:
            self.root.update_idletasks()
            self.root.update()

    def close(self) -> None:
        if self.root:
            self.root.destroy()

    def _read_float(self, value: str, name: str) -> float:
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"Invalid float for {name}: {value}") from exc

    def _read_int(self, value: str, name: str) -> int:
        try:
            return int(float(value))
        except ValueError as exc:
            raise ValueError(f"Invalid int for {name}: {value}") from exc

    def _collect_values(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, str, int, Path, bool]:
        root_pos = np.array(
            [self._read_float(v.get(), f"root_{i}") for i, v in enumerate(self.root_pos_vars)], dtype=np.float64
        )
        root_rpy = np.array(
            [self._read_float(v.get(), f"root_rpy_{i}") for i, v in enumerate(self.root_rpy_vars)], dtype=np.float64
        )
        joints = np.array(
            [self._read_float(v.get(), name) for v, name in zip(self.joint_vars, self.joint_names)],
            dtype=np.float64,
        )
        start = self._read_int(self.start_var.get(), "start")
        end = self._read_int(self.end_var.get(), "end")
        insert_frames = max(0, self._read_int(self.insert_var.get(), "insert"))
        mode = self.mode_var.get().strip()
        output_path = Path(self.output_var.get())
        apply_root = bool(self.apply_root_var.get())
        return root_pos, root_rpy, joints, start, end, mode, insert_frames, output_path, apply_root

    def _collect_range(self) -> tuple[int, int]:
        start = int(self._read_float(self.start_var.get(), "start"))
        end = int(self._read_float(self.end_var.get(), "end"))
        return start, end

    def _on_apply(self) -> None:
        try:
            root_pos, root_rpy, joints, start, end, mode, insert_frames, _out, apply_root = self._collect_values()
            self.apply_cb(root_pos, root_rpy, joints, start, end, mode, insert_frames, apply_root)
        except Exception as exc:  # noqa: BLE001
            self.messagebox.showerror("Apply error", str(exc))

    def _on_save(self) -> None:
        try:
            _root_pos, _root_rpy, _joints, _start, _end, _mode, _insert, out, _apply_root = self._collect_values()
            if not out.name:
                raise ValueError("Output path is empty.")
            self.save_cb(out)
        except Exception as exc:  # noqa: BLE001
            self.messagebox.showerror("Save error", str(exc))

    def _on_revert(self) -> None:
        try:
            start, end = self._collect_range()
            self.revert_cb(start, end)
        except Exception as exc:  # noqa: BLE001
            self.messagebox.showerror("Revert error", str(exc))

    def refresh_values(self, root_pos: np.ndarray, root_rpy: np.ndarray, joint_pos: np.ndarray) -> None:
        for i in range(3):
            self.root_pos_vars[i].set(f"{root_pos[i]:.6f}")
            self.root_rpy_vars[i].set(f"{root_rpy[i]:.6f}")
        for i in range(len(self.joint_vars)):
            self.joint_vars[i].set(f"{joint_pos[i]:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MuJoCo PKL motion editor.")
    parser.add_argument("--input", required=True, type=Path, help="Input pkl/npz path.")
    parser.add_argument("--model", type=Path, default=Path(DEFAULT_URDF), help="URDF/MJCF model path.")
    parser.add_argument(
        "--meshdir",
        type=Path,
        default=None,
        help="Override MJCF compiler meshdir (useful for XML models).",
    )
    parser.add_argument(
        "--ui_font_size",
        type=int,
        default=13,
        help="Editor UI font size.",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="auto",
        choices=["auto", "tgdex", "dex_evt"],
        help="Select joint list by robot type.",
    )
    parser.add_argument("--fps", type=float, default=None, help="Override FPS if not stored in file.")
    parser.add_argument(
        "--root_rot_format",
        type=str,
        default="xyzw",
        choices=["xyzw", "wxyz"],
        help="Quaternion format stored in pkl.",
    )
    parser.add_argument("--default_range", type=int, default=50, help="Default interpolation half-range in frames.")
    parser.add_argument("--output", type=Path, default=None, help="Default output path for saving.")
    return parser.parse_args()


def load_motion(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix in (".pkl", ".pickle"):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.startswith("numpy._core"):
                _ensure_numpy_core_alias()
                with open(path, "rb") as f:
                    return pickle.load(f)
            raise
    if path.suffix == ".npz":
        return dict(np.load(path, allow_pickle=True))
    raise ValueError(f"Unsupported file type: {path.suffix}")


def select_joint_names(robot: str, dof_count: int) -> list[str]:
    if robot == "auto":
        for name, joints in ROBOT_JOINTS.items():
            if len(joints) == dof_count:
                return joints
        raise ValueError(f"No joint list matches dof_pos columns: {dof_count}")
    joints = ROBOT_JOINTS[robot]
    if len(joints) != dof_count:
        raise ValueError(f"Joint list '{robot}' expects {len(joints)} dof, got {dof_count}")
    return joints


def main() -> None:
    args = parse_args()
    mujoco, glfw, tk_mod, messagebox_mod = _import_or_exit()

    data = load_motion(args.input)
    if "dof_pos" not in data or "root_pos" not in data or "root_rot" not in data:
        raise ValueError("Missing required keys: root_pos, root_rot, dof_pos")

    dof_pos = np.asarray(data["dof_pos"], dtype=np.float64)
    root_pos = np.asarray(data["root_pos"], dtype=np.float64)
    root_rot_raw = np.asarray(data["root_rot"], dtype=np.float64)

    if root_pos.shape[0] != dof_pos.shape[0] or root_rot_raw.shape[0] != dof_pos.shape[0]:
        raise ValueError("Frame count mismatch between root_pos/root_rot/dof_pos")

    fps = float(data.get("fps", args.fps or 30.0))
    total_frames = dof_pos.shape[0]

    joint_names = select_joint_names(args.robot, dof_pos.shape[1])

    if args.root_rot_format == "xyzw":
        root_rot_wxyz = np.stack([quat_xyzw_to_wxyz(q) for q in root_rot_raw], axis=0)
    else:
        root_rot_wxyz = root_rot_raw.copy()
    root_rot_xyzw = np.stack([quat_wxyz_to_xyzw(q) for q in root_rot_wxyz], axis=0)

    root_euler = np.stack([quat_wxyz_to_euler_xyz(q) for q in root_rot_wxyz], axis=0)
    orig_root_pos = root_pos.copy()
    orig_root_rot_xyzw = root_rot_xyzw.copy()
    orig_root_euler = root_euler.copy()
    orig_dof_pos = dof_pos.copy()

    model_path = args.model
    if model_path.suffix.lower() == ".urdf":
        resolved = _rewrite_urdf_mesh_paths(model_path)
        if resolved != model_path:
            print(f"[INFO] Using rewritten URDF: {resolved}")
        model_path = resolved
    elif model_path.suffix.lower() in (".xml", ".mjcf"):
        resolved = _rewrite_mjcf_meshdir(model_path, args.meshdir)
        if resolved != model_path:
            print(f"[INFO] Using rewritten MJCF: {resolved}")
        model_path = resolved
    model = mujoco.MjModel.from_xml_path(str(model_path))
    mj_data = mujoco.MjData(model)

    free_joints = [i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE]
    root_qpos_adr = model.jnt_qposadr[free_joints[0]] if free_joints else None
    if root_qpos_adr is None:
        print("[WARN] No free joint in model; root pose will not be visualized.", file=sys.stderr)

    joint_map = []
    missing = []
    for dof_idx, name in enumerate(joint_names):
        try:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_adr = model.jnt_qposadr[joint_id]
            joint_map.append((dof_idx, qpos_adr, name))
        except Exception:  # noqa: BLE001
            missing.append(name)

    if missing:
        print("[WARN] Missing joints in model:", ", ".join(missing), file=sys.stderr)

    def apply_frame(frame_idx: int) -> None:
        if root_qpos_adr is not None:
            mj_data.qpos[root_qpos_adr : root_qpos_adr + 3] = root_pos[frame_idx]
            print(f"current frame is:{frame_idx}")
            mj_data.qpos[root_qpos_adr + 3 : root_qpos_adr + 7] = root_rot_wxyz[frame_idx]
        for dof_idx, qpos_adr, _name in joint_map:
            mj_data.qpos[qpos_adr] = dof_pos[frame_idx, dof_idx]
        mujoco.mj_forward(model, mj_data)

    current_frame = 0
    playing = True
    last_time = time.time()

    def apply_edits(
        target_root_pos: np.ndarray,
        target_root_rpy: np.ndarray,
        target_joints: np.ndarray,
        start: int,
        end: int,
        mode: str,
        apply_root: bool,
    ) -> None:
        nonlocal root_pos, root_euler, root_rot_wxyz, root_rot_xyzw, dof_pos
        key = current_frame
        if not (0 <= start <= key <= end < total_frames):
            raise ValueError(f"Invalid range start={start} key={key} end={end}")
        if target_joints.shape[0] != dof_pos.shape[1]:
            raise ValueError("Joint count mismatch")

        if apply_root:
            apply_interpolation(root_pos, start, key, end, target_root_pos, mode)
            apply_interpolation(root_euler, start, key, end, target_root_rpy, mode)
        apply_interpolation(dof_pos, start, key, end, target_joints, mode)

        if apply_root:
            for i in range(start, end + 1):
                root_rot_wxyz[i] = euler_xyz_to_quat_wxyz(root_euler[i])
            root_rot_xyzw[start : end + 1] = np.stack(
                [quat_wxyz_to_xyzw(q) for q in root_rot_wxyz[start : end + 1]], axis=0
            )

        apply_frame(current_frame)

    editor: EditorWindow | None = None

    def revert_range(start: int, end: int) -> None:
        nonlocal root_pos, root_euler, root_rot_wxyz, root_rot_xyzw, dof_pos, editor
        if not (0 <= start <= end < total_frames):
            raise ValueError(f"Invalid range start={start} end={end}")
        root_pos[start : end + 1] = orig_root_pos[start : end + 1]
        dof_pos[start : end + 1] = orig_dof_pos[start : end + 1]
        root_rot_xyzw[start : end + 1] = orig_root_rot_xyzw[start : end + 1]
        root_rot_wxyz[start : end + 1] = np.stack(
            [quat_xyzw_to_wxyz(q) for q in root_rot_xyzw[start : end + 1]], axis=0
        )
        root_euler[start : end + 1] = np.stack(
            [quat_wxyz_to_euler_xyz(q) for q in root_rot_wxyz[start : end + 1]], axis=0
        )
        apply_frame(current_frame)
        if editor is not None and editor.visible:
            editor.refresh_values(root_pos[current_frame], root_euler[current_frame], dof_pos[current_frame])

    def save_motion(out_path: Path) -> None:
        output = dict(data)
        output["root_pos"] = root_pos
        output["root_rot"] = root_rot_xyzw
        output["dof_pos"] = dof_pos
        if out_path.suffix == ".npz":
            np.savez(out_path, **output)
        else:
            with open(out_path, "wb") as f:
                pickle.dump(output, f)
        print(f"[INFO] Saved: {out_path}")

    default_output = args.output or args.input.with_name(args.input.stem + ".edited.pkl")
    editor = EditorWindow(
        tk_mod,
        messagebox_mod,
        joint_names,
        apply_cb=apply_edits,
        revert_cb=revert_range,
        save_cb=save_motion,
        font_size=args.ui_font_size,
    )

    def open_editor() -> None:
        nonlocal playing
        playing = False
        start = max(0, current_frame - args.default_range)
        end = min(total_frames - 1, current_frame + args.default_range)
        editor.open(
            current_frame,
            root_pos[current_frame],
            root_euler[current_frame],
            dof_pos[current_frame],
            start,
            end,
            default_output,
        )

    def toggle_play() -> None:
        nonlocal playing, last_time
        playing = not playing
        last_time = time.time()

    def step_frames(delta: int) -> None:
        nonlocal current_frame
        current_frame = (current_frame + delta) % total_frames
        apply_frame(current_frame)

    key_events: "queue.Queue[tuple[int, int]]" = queue.Queue()
    scrub_dir = 0  # -1 for left, +1 for right
    scrub_rate = max(10.0, float(fps))  # frames per second when scrubbing
    scrub_hold_delay = 0.35
    scrub_last_step = time.time()
    scrub_last_event = 0.0
    scrub_hold_start = 0.0
    scrub_timeout = 0.25
    key_repeat_window = 0.2
    last_key = None
    last_key_time = 0.0

    # Accept both GLFW keycodes and MuJoCo keycodes.
    key_space = {glfw.KEY_SPACE, ord(" ")}
    key_enter = {glfw.KEY_ENTER, glfw.KEY_KP_ENTER, ord("\r"), ord("\n")}
    key_left = {glfw.KEY_LEFT}
    key_right = {glfw.KEY_RIGHT}
    mjt_key = getattr(mujoco, "mjtKey", None)
    if mjt_key is not None:
        for name, pool in (
            ("SPACE", key_space),
            ("ENTER", key_enter),
            ("LEFT", key_left),
            ("RIGHT", key_right),
        ):
            if hasattr(mjt_key, name):
                pool.add(int(getattr(mjt_key, name)))

    def handle_key(*args: object) -> None:
        key = None
        action = glfw.PRESS
        if len(args) == 1:
            key = int(args[0])
        elif len(args) == 4:
            key, _scancode, action, _mods = args
        elif len(args) == 5:
            _win, key, _scancode, action, _mods = args
        else:
            return
        if key is None or action not in (glfw.PRESS, glfw.REPEAT, glfw.RELEASE):
            return
        try:
            key_events.put_nowait((int(key), int(action)))
        except Exception:  # noqa: BLE001
            pass

    def process_key(key: int, action: int, now: float) -> None:
        nonlocal playing, scrub_dir, scrub_last_event, scrub_last_step, scrub_hold_start, last_key, last_key_time
        is_repeat = False
        if action == glfw.REPEAT:
            is_repeat = True
        elif action == glfw.PRESS and last_key == key and (now - last_key_time) <= key_repeat_window:
            is_repeat = True
        last_key = key
        last_key_time = now
        if key in key_space and action == glfw.PRESS:
            toggle_play()
            scrub_dir = 0
        elif key in key_right:
            if action == glfw.RELEASE:
                scrub_dir = 0
            else:
                playing = False
                if not is_repeat:
                    scrub_dir = 1
                    scrub_hold_start = now
                    scrub_last_event = now
                    scrub_last_step = now
                    step_frames(1)
                else:
                    scrub_last_event = now
        elif key in key_left:
            if action == glfw.RELEASE:
                scrub_dir = 0
            else:
                playing = False
                if not is_repeat:
                    scrub_dir = -1
                    scrub_hold_start = now
                    scrub_last_event = now
                    scrub_last_step = now
                    step_frames(-1)
                else:
                    scrub_last_event = now
        elif key in key_enter and action == glfw.PRESS:
            if not editor.visible:
                open_editor()

    def poll_scrub_keys(now: float, window: object | None) -> None:
        nonlocal playing, scrub_dir, scrub_last_event, scrub_last_step, scrub_hold_start
        if window is None:
            return
        try:
            left = glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS
            right = glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS
        except Exception:  # noqa: BLE001
            return
        if left and not right:
            if scrub_dir != -1:
                playing = False
                scrub_dir = -1
                scrub_hold_start = now
                scrub_last_step = now
            scrub_last_event = now
        elif right and not left:
            if scrub_dir != 1:
                playing = False
                scrub_dir = 1
                scrub_hold_start = now
                scrub_last_step = now
            scrub_last_event = now
        else:
            scrub_dir = 0

    apply_frame(current_frame)

    try:
        with mujoco.viewer.launch_passive(model, mj_data, key_callback=handle_key) as viewer:
            while viewer.is_running():
                now = time.time()
                if playing and now - last_time >= 1.0 / fps:
                    steps = int((now - last_time) * fps)
                    if steps > 0:
                        current_frame = (current_frame + steps) % total_frames
                        last_time += steps / fps
                        apply_frame(current_frame)
                while not key_events.empty():
                    key, action = key_events.get_nowait()
                    process_key(key, action, now)
                poll_scrub_keys(now, getattr(viewer, "_window", None))
                if scrub_dir != 0:
                    if getattr(viewer, "_window", None) is None and now - scrub_last_event > scrub_timeout:
                        scrub_dir = 0
                    elif now - scrub_hold_start >= scrub_hold_delay and now - scrub_last_step >= 1.0 / scrub_rate:
                        steps = int((now - scrub_last_step) * scrub_rate)
                        if steps > 0:
                            current_frame = (current_frame + scrub_dir * steps) % total_frames
                            scrub_last_step += steps / scrub_rate
                            apply_frame(current_frame)
                viewer.sync()
                editor.update()
    except TypeError:
        with mujoco.viewer.launch_passive(model, mj_data) as viewer:
            try:
                glfw.set_key_callback(viewer._window, handle_key)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                print("[WARN] Key callback not available; controls disabled.", file=sys.stderr)
            while viewer.is_running():
                now = time.time()
                if playing and now - last_time >= 1.0 / fps:
                    steps = int((now - last_time) * fps)
                    if steps > 0:
                        current_frame = (current_frame + steps) % total_frames
                        last_time += steps / fps
                        apply_frame(current_frame)
                while not key_events.empty():
                    key, action = key_events.get_nowait()
                    process_key(key, action, now)
                poll_scrub_keys(now, getattr(viewer, "_window", None))
                if scrub_dir != 0:
                    if getattr(viewer, "_window", None) is None and now - scrub_last_event > scrub_timeout:
                        scrub_dir = 0
                    elif now - scrub_hold_start >= scrub_hold_delay and now - scrub_last_step >= 1.0 / scrub_rate:
                        steps = int((now - scrub_last_step) * scrub_rate)
                        if steps > 0:
                            current_frame = (current_frame + scrub_dir * steps) % total_frames
                            scrub_last_step += steps / scrub_rate
                            apply_frame(current_frame)
                viewer.sync()
                editor.update()
    finally:
        editor.close()


if __name__ == "__main__":
    main()
