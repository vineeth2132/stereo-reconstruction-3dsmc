#!/usr/bin/env python3
"""Quantitative depth evaluation against the rectified ETH3D ground truth.

The C++ pipeline writes one metric depth map per dense-matching backend, tagged
``opencv`` and ``custom``:

    out/depth_<tag>_float.tiff   + out/valid_depth_<tag>_mask.tiff   (invalid = -1)

and one rectified ground-truth depth map (metric meters, invalid = NaN):

    out/gt_depth_rectified_float.tiff

For each tag this script intersects our-valid with GT-valid and reports coverage,
MAE, RMSE, median absolute error, and bad-pixel ratios, plus a median-scale-aligned
MAE that removes any global scale bias so it can be inspected separately. It prints a
compact comparison table and saves absolute-error maps (white = invalid, shared color
scale) mirroring scripts/visualize_maps.py:

    out/error_<tag>_scaled.png

Run with the project venv:

    .venv\\Scripts\\python.exe scripts\\evaluate_depth.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile
import cv2
import matplotlib
matplotlib.use("Agg")  # file output only, no display
import matplotlib.pyplot as plt


# Bad-pixel error thresholds in meters.
BAD_THRESHOLDS = (0.1, 0.5, 1.0)


def load_gt(out_dir: Path) -> np.ndarray | None:
    """Load the rectified GT depth as a float array with NaN at invalid pixels.
    Invalid GT is encoded as NaN (see DepthMapEvaluator::RectifyGroundTruthDepth);
    non-positive depths are treated as invalid too. Returns None if the file is
    missing."""
    gt_path = out_dir / "gt_depth_rectified_float.tiff"
    if not gt_path.exists():
        return None

    gt = tifffile.imread(gt_path).astype(np.float32)
    gt[~np.isfinite(gt)] = np.nan
    gt[gt <= 0.0] = np.nan
    return gt


def load_pred(out_dir: Path, tag: str) -> np.ndarray | None:
    """Load our metric depth map and its mask, returning a float array with NaN at
    invalid pixels (invalid = -1 with a separate mask tiff). Returns None if the
    value file is missing."""
    value_path = out_dir / f"depth_{tag}_float.tiff"
    mask_path = out_dir / f"valid_depth_{tag}_mask.tiff"

    if not value_path.exists():
        print(f"  skip {tag}: {value_path.name} not found")
        return None

    values = tifffile.imread(value_path).astype(np.float32)
    if mask_path.exists():
        mask = tifffile.imread(mask_path) > 0
    else:
        mask = np.isfinite(values) & (values > 0.0)

    out = values.copy()
    out[~mask] = np.nan
    out[~np.isfinite(out)] = np.nan
    return out

def align_gt_to_prediction(
    gt: np.ndarray,
    pred_shape: tuple[int, int],
    tag: str,
) -> np.ndarray:
    """Resize/crop GT so it matches the prediction resolution."""

    if gt.shape == pred_shape:
        return gt.copy()

    target_h, target_w = pred_shape

    if tag == "dust3r":
        gt_h, gt_w = gt.shape

        # DUSt3R scales the long edge, then center-crops vertically.
        scale = target_w / gt_w
        resized_h = int(round(gt_h * scale))

        resized = cv2.resize(
            gt.astype(np.float32),
            (target_w, resized_h),
            interpolation=cv2.INTER_NEAREST,
        )

        if resized_h < target_h:
            raise ValueError(
                f"DUSt3R aligned GT is too small: "
                f"{resized.shape}, prediction={pred_shape}"
            )

        crop_top = (resized_h - target_h) // 2
        aligned = resized[crop_top:crop_top + target_h, :]
    else:
        # RAFT and other lower-resolution predictions use direct resizing.
        aligned = cv2.resize(
            gt.astype(np.float32),
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        )

    if aligned.shape != pred_shape:
        raise ValueError(
            f"GT alignment shape mismatch: "
            f"got {aligned.shape}, expected {pred_shape}"
        )

    aligned = aligned.astype(np.float32, copy=False)
    aligned[~np.isfinite(aligned)] = np.nan
    aligned[aligned <= 0.0] = np.nan

    return aligned

def evaluate(
    pred: np.ndarray,
    gt: np.ndarray,
    tag: str,
) -> dict | None:
    """Compute direct and median-scale-aligned depth metrics.

    Direct metrics use the predicted metric depth unchanged.

    Scaled metrics first multiply the prediction by the median GT/prediction
    depth ratio. This removes one global scale bias while preserving the
    predicted depth structure.
    """

    gt_eval = align_gt_to_prediction(gt, pred.shape, tag)

    both = np.isfinite(pred) & np.isfinite(gt_eval)
    count = int(both.sum())
    gt_valid = int(np.isfinite(gt_eval).sum())

    if count == 0:
        return None

    p = pred[both].astype(np.float64)
    g = gt_eval[both].astype(np.float64)

    # Direct metric-depth errors.
    err = p - g
    abs_err = np.abs(err)

    # Median global scale alignment.
    valid_ratio = np.isfinite(g / p) & (p > 0.0)

    if not np.any(valid_ratio):
        return None

    scale = float(np.median(g[valid_ratio] / p[valid_ratio]))

    p_scaled = p * scale
    err_scaled = p_scaled - g
    abs_err_scaled = np.abs(err_scaled)

    metrics = {
        "count": count,
        "coverage": count / gt_valid if gt_valid else 0.0,
        "gt_valid": gt_valid,

        # Direct metrics.
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "median_ae": float(np.median(abs_err)),

        # Scale-aligned metrics.
        "scale": scale,
        "mae_scaled": float(abs_err_scaled.mean()),
        "rmse_scaled": float(np.sqrt(np.mean(err_scaled ** 2))),
        "median_ae_scaled": float(np.median(abs_err_scaled)),
    }

    for threshold in BAD_THRESHOLDS:
        metrics[f"bad_{threshold}"] = float(
            np.mean(abs_err > threshold)
        )

        metrics[f"bad_scaled_{threshold}"] = float(
            np.mean(abs_err_scaled > threshold)
        )

    return metrics

def print_table(results: dict[str, dict]) -> None:
    tags = list(results.keys())

    rows = [
        ("valid px", "count", "{:,}"),
        ("coverage", "coverage", "{:.1%}"),

        ("MAE (m)", "mae", "{:.4f}"),
        ("RMSE (m)", "rmse", "{:.4f}"),
        ("median AE (m)", "median_ae", "{:.4f}"),

        ("MAE scaled (m)", "mae_scaled", "{:.4f}"),
        ("RMSE scaled (m)", "rmse_scaled", "{:.4f}"),
        ("median AE scaled (m)", "median_ae_scaled", "{:.4f}"),

        ("scale (gt/pred)", "scale", "{:.4f}"),
    ]

    for threshold in BAD_THRESHOLDS:
        rows.append(
            (
                f"bad > {threshold}m",
                f"bad_{threshold}",
                "{:.1%}",
            )
        )

        rows.append(
            (
                f"bad scaled > {threshold}m",
                f"bad_scaled_{threshold}",
                "{:.1%}",
            )
        )

    label_w = max(len(label) for label, _, _ in rows)
    col_w = max(12, *(len(tag) for tag in tags))

    header = (
        "  "
        + " " * label_w
        + "  "
        + "  ".join(f"{tag:>{col_w}}" for tag in tags)
    )

    print(header)
    print("  " + "-" * (len(header) - 2))

    for label, key, fmt in rows:
        cells = []

        for tag in tags:
            value = results[tag].get(key)
            cells.append(fmt.format(value) if value is not None else "-")

        print(
            "  "
            + f"{label:<{label_w}}"
            + "  "
            + "  ".join(f"{cell:>{col_w}}" for cell in cells)
        )

def render_error_maps(
    pred_maps: dict[str, np.ndarray],
    gt: np.ndarray,
    out_dir: Path,
) -> None:
    """Save median-scale-aligned absolute-error maps.

    White pixels are invalid. All backends share one color scale.
    """

    error_maps: dict[str, np.ndarray] = {}

    for tag, pred in pred_maps.items():
        gt_eval = align_gt_to_prediction(gt, pred.shape, tag)

        both = np.isfinite(pred) & np.isfinite(gt_eval)

        err_map = np.full(pred.shape, np.nan, dtype=np.float32)

        if not np.any(both):
            error_maps[tag] = err_map
            continue

        p = pred[both].astype(np.float64)
        g = gt_eval[both].astype(np.float64)

        valid_ratio = np.isfinite(g / p) & (p > 0.0)

        if not np.any(valid_ratio):
            error_maps[tag] = err_map
            continue

        scale = float(np.median(g[valid_ratio] / p[valid_ratio]))

        err_map[both] = np.abs(
            pred[both].astype(np.float32) * np.float32(scale)
            - gt_eval[both]
        )

        error_maps[tag] = err_map

    finite_parts = [
        error[np.isfinite(error)].ravel()
        for error in error_maps.values()
        if np.any(np.isfinite(error))
    ]

    if not finite_parts:
        print("  no overlapping pixels to render error maps")
        return

    finite = np.concatenate(finite_parts)
    vmax = float(np.percentile(finite, 99))

    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("white")

    for tag, error in error_maps.items():
        fig, ax = plt.subplots(figsize=(9, 6))

        image = ax.imshow(
            error,
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
            interpolation="nearest",
        )

        ax.set_title(
            f"Scale-aligned absolute depth error — {tag} "
            "(white = invalid)"
        )
        ax.set_xticks([])
        ax.set_yticks([])

        colorbar = fig.colorbar(
            image,
            ax=ax,
            fraction=0.046,
            pad=0.04,
        )
        colorbar.set_label("|scaled error| (m)")

        fig.tight_layout()

        destination = out_dir / f"error_{tag}_scaled.png"
        fig.savefig(destination, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"  wrote {destination.name}")
        


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "out"),
                        help="Directory holding the .tiff maps (default: ../out)")
    parser.add_argument("--tags", nargs="+", default=["opencv", "custom"],
                        help="Backend tags to evaluate (default: opencv custom)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    if not out_dir.is_dir():
        raise SystemExit(f"Output directory not found: {out_dir}")

    gt = load_gt(out_dir)
    if gt is None:
        raise SystemExit(
            f"Ground-truth depth not found: {out_dir / 'gt_depth_rectified_float.tiff'}\n"
            "Run the pipeline with the ETH3D ground_truth_depth data downloaded to "
            "produce it, then re-run this script."
        )
    print(f"Loaded GT: {out_dir / 'gt_depth_rectified_float.tiff'} "
          f"({int(np.isfinite(gt).sum()):,} valid px)")

    pred_maps: dict[str, np.ndarray] = {}
    results: dict[str, dict] = {}
    for tag in args.tags:
        pred = load_pred(out_dir, tag)
        if pred is None:
            continue
        pred_maps[tag] = pred
        metrics = evaluate(pred, gt, tag)
        if metrics is None:
            print(f"  skip {tag}: no overlap with GT")
            continue
        results[tag] = metrics

    if not results:
        raise SystemExit("No backends could be evaluated (no overlap with GT).")

    print("\nDepth evaluation vs. rectified ETH3D ground truth:")
    print_table(results)

    print("\nError maps:")
    render_error_maps(pred_maps, gt, out_dir)


if __name__ == "__main__":
    main()
