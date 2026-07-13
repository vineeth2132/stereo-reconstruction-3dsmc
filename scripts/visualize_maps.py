#!/usr/bin/env python3
"""Render raw, filtered and filled disparity/depth maps with shared color scales.

Custom maps use:
    <kind>_<method>_<stage>_float.tiff
    valid_<kind>_<method>_<stage>_mask.tiff

OpenCV uses the single-result names:
    disparity_opencv_float.tiff
    valid_disparity_opencv_mask.tiff
    depth_opencv_final_float.tiff or depth_opencv_float.tiff
    valid_depth_opencv_final_mask.tiff or valid_depth_opencv_mask.tiff
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

CUSTOM_STAGES = ("raw", "filtered", "filled")


def read_masked(value_path: Path, mask_path: Path, *, positive_only: bool) -> np.ndarray | None:
    if not value_path.exists():
        return None

    values = tifffile.imread(value_path).astype(np.float32)
    if mask_path.exists():
        mask = tifffile.imread(mask_path) > 0
    else:
        mask = np.isfinite(values)
        if positive_only:
            mask &= values > 0.0

    result = values.copy()
    result[~mask] = np.nan
    result[~np.isfinite(result)] = np.nan
    if positive_only:
        result[result <= 0.0] = np.nan
    return result


def load_stage_map(out_dir: Path, kind: str, method: str, stage: str) -> np.ndarray | None:
    return read_masked(out_dir / f"{kind}_{method}_{stage}_float.tiff", out_dir / f"valid_{kind}_{method}_{stage}_mask.tiff", positive_only=(kind == "depth"))


def load_opencv_map(out_dir: Path, kind: str) -> np.ndarray | None:
    candidates = [
        (out_dir / f"{kind}_opencv_final_float.tiff", out_dir / f"valid_{kind}_opencv_final_mask.tiff"),
        (out_dir / f"{kind}_opencv_float.tiff", out_dir / f"valid_{kind}_opencv_mask.tiff"),
    ]
    for value_path, mask_path in candidates:
        result = read_masked(value_path, mask_path, positive_only=(kind == "depth"))
        if result is not None:
            return result
    return None


def load_gt_depth(out_dir: Path) -> np.ndarray | None:
    path = out_dir / "gt_depth_rectified_float.tiff"
    if not path.exists():
        return None

    gt = tifffile.imread(path).astype(np.float32)
    gt[~np.isfinite(gt)] = np.nan
    gt[gt <= 0.0] = np.nan
    return gt


def shared_limits(maps: list[np.ndarray]) -> tuple[float, float]:
    finite_parts = [data[np.isfinite(data)] for data in maps if np.isfinite(data).any()]
    if not finite_parts:
        return 0.0, 1.0
    finite = np.concatenate(finite_parts)
    return float(np.percentile(finite, 1)), float(np.percentile(finite, 99))


def render_single(data: np.ndarray, title: str, label: str, destination: Path, limits: tuple[float, float]) -> None:
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(9, 6))
    image = ax.imshow(data, cmap=cmap, vmin=limits[0], vmax=limits[1], interpolation="nearest")
    ax.set_title(f"{title}  (valid: {100.0 * np.isfinite(data).mean():.1f}% of image px)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04).set_label(label)
    fig.tight_layout()
    fig.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {destination.name}")


def render_comparison(panels: list[tuple[str, np.ndarray]], label: str, destination: Path) -> None:
    if not panels:
        return

    limits = shared_limits([data for _, data in panels])
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("white")
    fig, axes = plt.subplots(1, len(panels), figsize=(8 * len(panels), 6), squeeze=False)

    image = None
    for ax, (title, data) in zip(axes[0], panels):
        image = ax.imshow(data, cmap=cmap, vmin=limits[0], vmax=limits[1], interpolation="nearest")
        ax.set_title(f"{title}\nvalid: {100.0 * np.isfinite(data).mean():.1f}%")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(image, ax=axes[0].tolist(), fraction=0.035, pad=0.03).set_label(label)
    fig.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {destination.name}")


def derive_gt_disparity(gt_depth: np.ndarray, disparity_maps: dict[str, np.ndarray], depth_maps: dict[str, np.ndarray]) -> np.ndarray | None:
    for key, disparity in disparity_maps.items():
        depth = depth_maps.get(key)
        if depth is None:
            continue
        both = np.isfinite(disparity) & np.isfinite(depth)
        if not both.any():
            continue

        f_b = float(np.median(disparity[both].astype(np.float64) * depth[both].astype(np.float64)))
        result = np.full(gt_depth.shape, np.nan, dtype=np.float32)
        valid = np.isfinite(gt_depth)
        result[valid] = f_b / gt_depth[valid]
        print(f"  derived fB from {key}: {f_b:.4f}")
        return result
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "out"))
    parser.add_argument("--methods", nargs="+", default=["NCC", "SSD", "Census"])
    parser.add_argument("--stages", nargs="+", choices=CUSTOM_STAGES, default=list(CUSTOM_STAGES))
    args = parser.parse_args()

    out_dir = Path(args.out)
    if not out_dir.is_dir():
        raise SystemExit(f"Output directory not found: {out_dir}")

    gt_depth = load_gt_depth(out_dir)

    for kind, colorbar_label in (("disparity", "Disparity (px)"), ("depth", "Depth Z (m)")):
        loaded: dict[str, np.ndarray] = {}

        opencv = load_opencv_map(out_dir, kind)
        if opencv is not None:
            loaded["opencv-final"] = opencv

        for method in args.methods:
            for stage in args.stages:
                key = f"{method}-{stage}"
                data = load_stage_map(out_dir, kind, method, stage)
                if data is not None:
                    loaded[key] = data

        if not loaded:
            print(f"{kind}: no maps found")
            continue

        limits = shared_limits(list(loaded.values()))
        print(f"{kind}:")
        for key, data in loaded.items():
            safe_key = key.replace("-", "_")
            render_single(data, f"{kind.capitalize()} — {key}", colorbar_label, out_dir / f"{kind}_{safe_key}_scaled.png", limits)

        for stage in args.stages:
            panels: list[tuple[str, np.ndarray]] = []
            if "opencv-final" in loaded:
                panels.append(("OpenCV-final", loaded["opencv-final"]))
            panels.extend((method, loaded[f"{method}-{stage}"]) for method in args.methods if f"{method}-{stage}" in loaded)

            if kind == "depth" and gt_depth is not None:
                panels.append(("Ground truth", gt_depth))
            elif kind == "disparity" and gt_depth is not None:
                disparity_maps = {key: value for key, value in loaded.items() if key != "opencv-final"}
                depth_maps = {f"{method}-{stage}": load_stage_map(out_dir, "depth", method, stage) for method in args.methods}
                gt_disparity = derive_gt_disparity(gt_depth, disparity_maps, depth_maps)
                if gt_disparity is not None:
                    panels.append(("Ground truth", gt_disparity))

            render_comparison(panels, colorbar_label, out_dir / f"comparison_{kind}_{stage}.png")


if __name__ == "__main__":
    main()
