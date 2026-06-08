"""Split an npz motion file into two parts at a specific frame.

Usage
-----
python scripts/split_npz.py --input path/to/motion.npz --split_frame 300 \\
    --output_dir ./out --output_prefix motion

This will emit:
- ./out/motion_part1_300.npz  (frames [0, split_frame))
- ./out/motion_part2_300.npz  (frames [split_frame, end))
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np


def _load_npz(path: Path) -> dict[str, Any]:
    """Load an npz/pkl file and return a dictionary-like payload."""
    # handle legacy pickle files saved via np.savez with allow_pickle=True
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.lib.npyio.NpzFile):
        data = dict(obj.items())
        obj.close()
        return data
    if isinstance(obj, np.ndarray):
        try:
            data = obj.item()
            if not isinstance(data, dict):
                raise TypeError("文件内对象不是字典")
            return data
        except Exception as exc:  # noqa: BLE001
            raise TypeError(f"无法从 {path} 解析出字典，请确认文件是npz或包含字典的pkl") from exc
    raise TypeError(f"不支持的文件类型: {type(obj)}")


def _infer_frame_count(data: dict[str, Any], frame_key: str | None) -> tuple[int, str]:
    """Infer total frame count and the key used for inference."""
    if frame_key is not None:
        frames = int(np.asarray(data[frame_key]).shape[0])
        return frames, frame_key
    # try common keys
    for key in ("joint_pos", "root_pos", "body_pos_w"):
        if key in data:
            arr = np.asarray(data[key])
            if arr.shape:
                return int(arr.shape[0]), key
    # fallback: first array-like with a length
    for key, val in data.items():
        arr = np.asarray(val)
        if arr.shape:
            return int(arr.shape[0]), key
    raise ValueError("无法推断帧数：没有找到可用的数组键")


def split_npz(
    input_path: Path,
    split_frame: int,
    output_dir: Path,
    output_prefix: str | None = None,
    frame_key: str | None = None,
) -> tuple[Path, Path]:
    """Split npz data by frame and save two files."""
    data = _load_npz(input_path)
    total_frames, used_key = _infer_frame_count(data, frame_key)
    if not 0 < split_frame < total_frames:
        raise ValueError(f"split_frame 必须在 (0, {total_frames}) 之间，当前为 {split_frame}")

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_prefix or input_path.stem
    out1 = output_dir / f"{prefix}_part1_{split_frame}.npz"
    out2 = output_dir / f"{prefix}_part2_{split_frame}.npz"

    head, tail = {}, {}
    for key, val in data.items():
        arr = np.asarray(val)
        if arr.shape and arr.shape[0] == total_frames:
            head[key] = arr[:split_frame]
            tail[key] = arr[split_frame:]
        else:
            # 非时间序列字段（如fps、单个标量）直接复制
            head[key] = val
            tail[key] = val

    np.savez(out1, **head)
    np.savez(out2, **tail)
    print(f"[INFO] 使用键 '{used_key}' 推断帧数 {total_frames}，在 {split_frame} 处分割。")
    print(f"[INFO] 保存: {out1}")
    print(f"[INFO] 保存: {out2}")
    return out1, out2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split an npz motion file by frame index.")
    parser.add_argument("--input", required=True, type=Path, help="输入 npz/pkl 文件路径")
    parser.add_argument("--split_frame", required=True, type=int, help="在该帧索引处分割")
    parser.add_argument("--output_dir", type=Path, default=Path("./motion_split"), help="输出目录")
    parser.add_argument("--output_prefix", type=str, default=None, help="输出文件名前缀，默认沿用输入文件名")
    parser.add_argument(
        "--frame_key",
        type=str,
        default=None,
        help="用于推断帧数的键（可选，如 joint_pos/root_pos），不填则自动寻找",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_npz(
        input_path=args.input,
        split_frame=args.split_frame,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        frame_key=args.frame_key,
    )


if __name__ == "__main__":
    main()
