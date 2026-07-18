#!/usr/bin/env python3
"""Run RAFT-Stereo on a rectified stereo pair and save metric depth output.

Expected input files:

    out/rectified_left.png
    out/rectified_right.png

Expected RAFT-Stereo repository:

    third_party/RAFT-Stereo/

Expected checkpoint:

    third_party/RAFT-Stereo/models/raftstereo-eth3d.pth

Generated output files:

    out/disparity_raft_float.tiff
    out/depth_raft_float.tiff
    out/valid_depth_raft_mask.tiff

Example:

    python scripts/run_raft_depth.py --width 640

Higher-resolution example:

    python scripts/run_raft_depth.py --width 2048 --mixed-precision

Use --width 0 to attempt inference at the original resolution.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import tifffile


DEFAULT_FX_PIXELS = 3408.59
DEFAULT_BASELINE_METERS = 0.4289


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--raft-root",
        type=Path,
        default=project_root / "third_party" / "RAFT-Stereo",
        help="Path to the cloned official RAFT-Stereo repository.",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            project_root
            / "third_party"
            / "RAFT-Stereo"
            / "models"
            / "raftstereo-eth3d.pth"
        ),
        help="Path to the RAFT-Stereo checkpoint.",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=project_root / "out",
        help=(
            "Directory containing rectified_left.png and "
            "rectified_right.png and receiving RAFT outputs."
        ),
    )

    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help=(
            "Inference width. Height is calculated while preserving the "
            "aspect ratio. Both dimensions are rounded down to multiples "
            "of 32. Use 0 for original resolution."
        ),
    )

    parser.add_argument(
        "--fx",
        type=float,
        default=None,
        help=(
            "Rectified-frame horizontal focal length in pixels. If omitted, "
            "the value is read from out/rectified_meta.txt (written by the "
            f"C++ pipeline), falling back to {DEFAULT_FX_PIXELS}."
        ),
    )

    parser.add_argument(
        "--baseline",
        type=float,
        default=None,
        help=(
            "Physical stereo baseline in meters. If omitted, read from "
            "out/rectified_meta.txt (written by the C++ pipeline), falling "
            f"back to {DEFAULT_BASELINE_METERS}."
        ),
    )

    parser.add_argument(
        "--valid-iters",
        type=int,
        default=8,
        help="Number of recurrent RAFT refinement iterations.",
    )

    parser.add_argument(
        "--corr-implementation",
        default="alt",
        choices=("reg", "alt", "reg_cuda", "alt_cuda"),
        help="RAFT correlation implementation.",
    )

    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable mixed-precision inference to reduce GPU memory usage.",
    )

    return parser.parse_args()


def read_rectified_meta(out_dir: Path) -> dict[str, float]:
    """Parse out/rectified_meta.txt (written by the C++ pipeline):
    'key value' per line, e.g. fx / baseline / width / height."""

    meta_path = out_dir / "rectified_meta.txt"
    meta: dict[str, float] = {}
    if meta_path.exists():
        for line in meta_path.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                try:
                    meta[parts[0]] = float(parts[1])
                except ValueError:
                    pass
    return meta


def resolve_camera_parameters(args: argparse.Namespace) -> None:
    """Resolve fx and baseline: explicit flags win, then the
    rectified_meta.txt written by the C++ pipeline, then the historical
    delivery_area defaults. The rectified fx differs from the raw
    cameras.txt fx, and both fx and baseline change per scene/pair, so
    the meta file is the reliable source."""

    meta = read_rectified_meta(args.out)

    if args.fx is None:
        if "fx" in meta:
            args.fx = meta["fx"]
            print(f"Using rectified fx from rectified_meta.txt: {args.fx:.4f} px")
        else:
            args.fx = DEFAULT_FX_PIXELS
            print(
                "Warning: rectified_meta.txt not found; falling back to the "
                f"delivery_area default fx = {DEFAULT_FX_PIXELS} px (raw-frame "
                "value; metric depth may carry a global scale bias)."
            )

    if args.baseline is None:
        if "baseline" in meta:
            args.baseline = meta["baseline"]
            print(f"Using baseline from rectified_meta.txt: {args.baseline:.6f} m")
        else:
            args.baseline = DEFAULT_BASELINE_METERS
            print(
                "Warning: rectified_meta.txt not found; falling back to the "
                f"delivery_area default baseline = {DEFAULT_BASELINE_METERS} m."
            )


def validate_arguments(args: argparse.Namespace) -> None:
    if args.width < 0:
        raise SystemExit("--width must be zero or a positive integer")

    if args.fx <= 0:
        raise SystemExit("--fx must be positive")

    if args.baseline <= 0:
        raise SystemExit("--baseline must be positive")

    if args.valid_iters <= 0:
        raise SystemExit("--valid-iters must be positive")


def resize_pair(
    left: np.ndarray,
    right: np.ndarray,
    requested_width: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Resize both images identically and return the horizontal scale factor."""

    if left.shape[:2] != right.shape[:2]:
        raise ValueError(
            "Left and right image dimensions differ: "
            f"{left.shape[:2]} vs {right.shape[:2]}"
        )

    original_height, original_width = left.shape[:2]

    if requested_width == 0:
        target_width = (original_width // 32) * 32
    else:
        target_width = (requested_width // 32) * 32

    if target_width < 32:
        raise ValueError("Inference width must be at least 32 pixels")

    scale = target_width / original_width

    target_height = int(round(original_height * scale))
    target_height = (target_height // 32) * 32

    if target_height < 32:
        raise ValueError("Calculated inference height is smaller than 32 pixels")

    if target_width == original_width and target_height == original_height:
        return left.copy(), right.copy(), 1.0

    left_resized = cv2.resize(
        left,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )

    right_resized = cv2.resize(
        right,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )

    actual_scale = target_width / original_width

    return left_resized, right_resized, actual_scale


def normalize_disparity(disparity: np.ndarray) -> np.ndarray:
    """Convert common RAFT output shapes into one HxW float disparity map."""

    disparity = np.asarray(disparity, dtype=np.float32)
    disparity = np.squeeze(disparity)

    if disparity.ndim == 3:
        # Handle possible [2, H, W] or [H, W, 2] optical-flow-style output.
        if disparity.shape[0] in (1, 2):
            disparity = disparity[0]
        elif disparity.shape[-1] in (1, 2):
            disparity = disparity[..., 0]

    if disparity.ndim != 2:
        raise ValueError(
            f"Unexpected RAFT disparity shape after squeezing: {disparity.shape}"
        )

    # RAFT-Stereo may represent leftward flow with a negative sign.
    return np.abs(disparity).astype(np.float32)


def find_generated_npy(demo_output: Path) -> Path:
    npy_files = list(demo_output.glob("*.npy"))

    if not npy_files:
        raise FileNotFoundError(
            f"RAFT inference completed but no .npy file was found in {demo_output}"
        )

    # Normally only one file exists. Selecting newest is robust if more are present.
    return max(npy_files, key=lambda path: path.stat().st_mtime)


def main() -> None:
    args = parse_args()
    resolve_camera_parameters(args)
    validate_arguments(args)

    raft_root = args.raft_root.resolve()
    checkpoint = args.checkpoint.resolve()
    out_dir = args.out.resolve()

    demo_script = raft_root / "demo.py"
    left_path = out_dir / "rectified_left.png"
    right_path = out_dir / "rectified_right.png"

    required_paths = (
        raft_root,
        demo_script,
        checkpoint,
        left_path,
        right_path,
    )

    for path in required_paths:
        if not path.exists():
            raise SystemExit(f"Required path not found: {path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
    right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)

    if left is None:
        raise SystemExit(f"Could not read left image: {left_path}")

    if right is None:
        raise SystemExit(f"Could not read right image: {right_path}")

    original_height, original_width = left.shape[:2]

    left_input, right_input, image_scale = resize_pair(
        left,
        right,
        args.width,
    )

    input_height, input_width = left_input.shape[:2]
    scaled_fx = args.fx * image_scale

    raft_input_dir = out_dir / "raft_input"
    raft_input_dir.mkdir(parents=True, exist_ok=True)

    raft_left_path = raft_input_dir / "rectified_left_raft.png"
    raft_right_path = raft_input_dir / "rectified_right_raft.png"

    if not cv2.imwrite(str(raft_left_path), left_input):
        raise SystemExit(f"Could not write RAFT input: {raft_left_path}")

    if not cv2.imwrite(str(raft_right_path), right_input):
        raise SystemExit(f"Could not write RAFT input: {raft_right_path}")

    demo_output = raft_root / "demo_output"

    # Remove old demo outputs so the wrapper always knows which result is new.
    if demo_output.exists():
        shutil.rmtree(demo_output)

    command = [
        sys.executable,
        str(demo_script),
        "--restore_ckpt",
        checkpoint.as_posix(),
        "-l",
        raft_left_path.as_posix(),
        "-r",
        raft_right_path.as_posix(),
        "--save_numpy",
        "--corr_implementation",
        args.corr_implementation,
        "--valid_iters",
        str(args.valid_iters),
    ]

    if args.mixed_precision:
        command.append("--mixed_precision")

    print("Running RAFT-Stereo")
    print("-------------------")
    print(f"Original resolution: {original_width}x{original_height}")
    print(f"RAFT resolution:     {input_width}x{input_height}")
    print(f"Original fx:         {args.fx:.4f} px")
    print(f"Scaled fx:           {scaled_fx:.4f} px")
    print(f"Baseline:            {args.baseline:.6f} m")
    print(f"Checkpoint:          {checkpoint}")
    print()
    print("Command:")
    print(" ".join(command))
    print()

    try:
        subprocess.run(
            command,
            cwd=raft_root,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            f"RAFT-Stereo inference failed with exit code {error.returncode}"
        ) from error

    disparity_npy = find_generated_npy(demo_output)
    disparity = normalize_disparity(np.load(disparity_npy))

    if disparity.shape != (input_height, input_width):
        raise SystemExit(
            "RAFT output resolution differs from expected input resolution: "
            f"output={disparity.shape}, expected={(input_height, input_width)}"
        )

    valid_disparity = (
        np.isfinite(disparity)
        & (disparity > 1e-6)
    )

    depth = np.full(disparity.shape, -1.0, dtype=np.float32)

    depth[valid_disparity] = (
        np.float32(scaled_fx * args.baseline)
        / disparity[valid_disparity]
    )

    valid_depth = (
        valid_disparity
        & np.isfinite(depth)
        & (depth > 0.0)
    )

    mask = valid_depth.astype(np.uint8) * 255

    disparity_output = np.full(disparity.shape, -1.0, dtype=np.float32)
    disparity_output[valid_disparity] = disparity[valid_disparity]

    disparity_path = out_dir / "disparity_raft_float.tiff"
    depth_path = out_dir / "depth_raft_float.tiff"
    mask_path = out_dir / "valid_depth_raft_mask.tiff"
    npy_output_path = out_dir / "disparity_raft.npy"

    tifffile.imwrite(
        disparity_path,
        disparity_output.astype(np.float32),
    )

    tifffile.imwrite(
        depth_path,
        depth.astype(np.float32),
    )

    tifffile.imwrite(
        mask_path,
        mask.astype(np.uint8),
    )

    np.save(
        npy_output_path,
        disparity.astype(np.float32),
    )

    valid_depth_values = depth[valid_depth]

    print()
    print("RAFT-Stereo depth completed")
    print("---------------------------")
    print(f"Original input:       {original_width}x{original_height}")
    print(f"RAFT input/output:    {input_width}x{input_height}")
    print(f"Image scale:          {image_scale:.6f}")
    print(f"Scaled focal length:  {scaled_fx:.4f} px")
    print(f"Physical baseline:    {args.baseline:.6f} m")
    print(
        f"Valid pixels:         {int(valid_depth.sum()):,} / "
        f"{valid_depth.size:,}"
    )

    if valid_depth_values.size:
        print(
            "Metric depth range:    "
            f"{float(valid_depth_values.min()):.4f} to "
            f"{float(valid_depth_values.max()):.4f} m"
        )
        print(
            "Median metric depth:   "
            f"{float(np.median(valid_depth_values)):.4f} m"
        )

    print(f"Wrote disparity:      {disparity_path}")
    print(f"Wrote depth:          {depth_path}")
    print(f"Wrote valid mask:     {mask_path}")
    print(f"Wrote NumPy output:   {npy_output_path}")


if __name__ == "__main__":
    main()