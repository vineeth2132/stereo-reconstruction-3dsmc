#!/usr/bin/env python3
"""Evaluate raw, filtered and filled stereo outputs against rectified ETH3D depth GT.

Expected custom output names:
    disparity_<method>_<stage>_float.tiff
    valid_disparity_<method>_<stage>_mask.tiff
    depth_<method>_<stage>_float.tiff
    valid_depth_<method>_<stage>_mask.tiff

where method is typically NCC, SSD or Census and stage is raw, filtered or filled.

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

BAD_THRESHOLDS = (0.1, 0.5, 1.0)
BAD_DISP_THRESHOLDS = (1.0, 2.0, 4.0)
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


def load_gt(out_dir: Path) -> np.ndarray:
    path = out_dir / "gt_depth_rectified_float.tiff"
    if not path.exists():
        raise SystemExit(f"Ground-truth depth not found: {path}")

    gt = tifffile.imread(path).astype(np.float32)
    gt[~np.isfinite(gt)] = np.nan
    gt[gt <= 0.0] = np.nan
    return gt


def load_stage_map(out_dir: Path, kind: str, method: str, stage: str) -> np.ndarray | None:
    value_path = out_dir / f"{kind}_{method}_{stage}_float.tiff"
    mask_path = out_dir / f"valid_{kind}_{method}_{stage}_mask.tiff"
    return read_masked(value_path, mask_path, positive_only=(kind == "depth"))


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


def evaluate_depth(pred: np.ndarray, gt: np.ndarray) -> dict[str, float | int] | None:
    both = np.isfinite(pred) & np.isfinite(gt)
    count = int(both.sum())
    gt_valid = int(np.isfinite(gt).sum())
    if count == 0:
        return None

    p = pred[both].astype(np.float64)
    g = gt[both].astype(np.float64)
    err = p - g
    abs_err = np.abs(err)

    valid_scale = np.abs(p) > 1e-12
    scale = float(np.median(g[valid_scale] / p[valid_scale])) if valid_scale.any() else float("nan")
    mae_scaled = float(np.mean(np.abs(p * scale - g))) if np.isfinite(scale) else float("nan")

    metrics: dict[str, float | int] = {
        "count": count,
        "coverage": count / gt_valid if gt_valid else 0.0,
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "median_ae": float(np.median(abs_err)),
        "scale": scale,
        "mae_scaled": mae_scaled,
    }
    for threshold in BAD_THRESHOLDS:
        metrics[f"bad_{threshold}"] = float(np.mean(abs_err > threshold))
    return metrics


def add_disparity_metrics(metrics: dict[str, float | int], disp: np.ndarray, depth: np.ndarray, gt: np.ndarray) -> None:
    own = np.isfinite(disp) & np.isfinite(depth)
    if not own.any():
        return

    f_b = float(np.median(disp[own].astype(np.float64) * depth[own].astype(np.float64)))
    gt_disp = np.full(gt.shape, np.nan, dtype=np.float64)
    gt_valid = np.isfinite(gt)
    gt_disp[gt_valid] = f_b / gt[gt_valid]

    both = np.isfinite(disp) & np.isfinite(gt_disp)
    metrics["fB"] = f_b
    if not both.any():
        return

    abs_err = np.abs(disp[both].astype(np.float64) - gt_disp[both])
    for threshold in BAD_DISP_THRESHOLDS:
        metrics[f"bad_disp_{threshold}"] = float(np.mean(abs_err > threshold))


def print_table(results: dict[str, dict[str, float | int]]) -> None:
    rows = [
        ("valid px", "count", "{:,}"),
        ("coverage", "coverage", "{:.1%}"),
        ("MAE (m)", "mae", "{:.4f}"),
        ("RMSE (m)", "rmse", "{:.4f}"),
        ("median AE (m)", "median_ae", "{:.4f}"),
        ("MAE scaled (m)", "mae_scaled", "{:.4f}"),
        ("scale (gt/pred)", "scale", "{:.4f}"),
    ]
    rows.extend((f"bad > {t}m", f"bad_{t}", "{:.1%}") for t in BAD_THRESHOLDS)
    rows.extend((f"bad > {int(t)}px", f"bad_disp_{t}", "{:.1%}") for t in BAD_DISP_THRESHOLDS)

    labels = list(results)
    label_width = max(len(label) for label, _, _ in rows)
    column_width = max(16, *(len(label) for label in labels))
    header = "  " + " " * label_width + "  " + "  ".join(f"{label:>{column_width}}" for label in labels)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for label, key, fmt in rows:
        cells = []
        for result in results.values():
            value = result.get(key)
            cells.append(fmt.format(value) if value is not None else "-")
        print("  " + f"{label:<{label_width}}" + "  " + "  ".join(f"{cell:>{column_width}}" for cell in cells))


def render_error_maps(predictions: dict[str, np.ndarray], gt: np.ndarray, out_dir: Path) -> None:
    errors: dict[str, np.ndarray] = {}
    finite_parts: list[np.ndarray] = []

    for label, pred in predictions.items():
        both = np.isfinite(pred) & np.isfinite(gt)
        error = np.full(pred.shape, np.nan, dtype=np.float32)
        error[both] = np.abs(pred[both] - gt[both])
        errors[label] = error
        finite = error[np.isfinite(error)]
        if finite.size:
            finite_parts.append(finite)

    if not finite_parts:
        return

    vmax = float(np.percentile(np.concatenate(finite_parts), 99))
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("white")

    for label, error in errors.items():
        fig, ax = plt.subplots(figsize=(9, 6))
        image = ax.imshow(error, cmap=cmap, vmin=0.0, vmax=vmax, interpolation="nearest")
        ax.set_title(f"Absolute depth error — {label}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04).set_label("|error| (m)")
        fig.tight_layout()
        safe_label = label.replace("/", "_").replace(" ", "_")
        destination = out_dir / f"error_{safe_label}_scaled.png"
        fig.savefig(destination, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {destination.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "out"))
    parser.add_argument("--methods", nargs="+", default=["NCC", "SSD", "Census"])
    parser.add_argument("--stages", nargs="+", choices=CUSTOM_STAGES, default=list(CUSTOM_STAGES))
    args = parser.parse_args()

    out_dir = Path(args.out)
    if not out_dir.is_dir():
        raise SystemExit(f"Output directory not found: {out_dir}")

    gt = load_gt(out_dir)
    print(f"Loaded GT with {int(np.isfinite(gt).sum()):,} valid pixels")

    results: dict[str, dict[str, float | int]] = {}
    predictions: dict[str, np.ndarray] = {}

    depth = load_opencv_map(out_dir, "depth")
    if depth is not None:
        metrics = evaluate_depth(depth, gt)
        if metrics is not None:
            disparity = load_opencv_map(out_dir, "disparity")
            if disparity is not None:
                add_disparity_metrics(metrics, disparity, depth, gt)
            results["opencv-final"] = metrics
            predictions["opencv-final"] = depth
    else:
        print("  skip opencv-final: depth map not found")

    for method in args.methods:
        for stage in args.stages:
            label = f"{method}-{stage}"
            depth = load_stage_map(out_dir, "depth", method, stage)
            if depth is None:
                print(f"  skip {label}: depth map not found")
                continue

            metrics = evaluate_depth(depth, gt)
            if metrics is None:
                print(f"  skip {label}: no overlap with GT")
                continue

            disparity = load_stage_map(out_dir, "disparity", method, stage)
            if disparity is not None:
                add_disparity_metrics(metrics, disparity, depth, gt)

            results[label] = metrics
            predictions[label] = depth

    if not results:
        raise SystemExit("No outputs could be evaluated.")

    print("\nDepth evaluation vs. rectified ETH3D ground truth:")
    print_table(results)

    print("\nError maps:")
    render_error_maps(predictions, gt, out_dir)


if __name__ == "__main__":
    main()
