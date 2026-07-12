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
    """Align ground truth with a backend prediction.

    RAFT and other lower-resolution outputs use a direct resize. DUSt3R first
    resizes the image so its long edge is 512 pixels, then center-crops the
    result to dimensions compatible with the model patch size.
    """
    if gt.shape == pred_shape:
        return gt

    target_h, target_w = pred_shape

    if tag == "dust3r":
        gt_h, gt_w = gt.shape

        # DUSt3R load_images(..., size=512) scales the long image edge to 512.
        scale = target_w / gt_w
        resized_h = int(round(gt_h * scale))

        resized = cv2.resize(
            gt.astype(np.float32),
            (target_w, resized_h),
            interpolation=cv2.INTER_NEAREST,
        )

        if resized_h < target_h:
            raise ValueError(
                "DUSt3R GT alignment produced an image smaller than the "
                f"prediction: resized={resized.shape}, prediction={pred_shape}"
            )

        crop_top = (resized_h - target_h) // 2
        aligned = resized[crop_top:crop_top + target_h, :]

        if aligned.shape != pred_shape:
            raise ValueError(
                "Unexpected DUSt3R GT alignment shape: "
                f"{aligned.shape}, expected {pred_shape}"
            )
    else:
        # Preserve the existing RAFT/general lower-resolution behavior.
        aligned = cv2.resize(
            gt.astype(np.float32),
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
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
    """Compute error metrics over the pixels valid in both maps. Returns None if
    there is no overlap."""

    gt_eval = align_gt_to_prediction(gt, pred.shape, tag)

    both = np.isfinite(pred) & np.isfinite(gt_eval)
    count = int(both.sum())
    gt_valid = int(np.isfinite(gt_eval).sum())

    if count == 0:
        return None

    p = pred[both].astype(np.float64)
    g = gt_eval[both].astype(np.float64)
    err = p - g
    abs_err = np.abs(err)

    # Global scale bias: align predictions to GT by the median depth ratio, then
    # re-measure MAE so a pure scale error shows up separately from structural error.
    scale = float(np.median(g / p))
    abs_err_scaled = np.abs(p * scale - g)

    metrics = {
        "count": count,
        "coverage": count / gt_valid if gt_valid else 0.0,
        "gt_valid": gt_valid,
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "median_ae": float(np.median(abs_err)),
        "scale": scale,
        "mae_scaled": float(abs_err_scaled.mean()),
    }

    for t in BAD_THRESHOLDS:
        metrics[f"bad_{t}"] = float((abs_err > t).mean())

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
        ("scale (gt/pred)", "scale", "{:.4f}"),
    ]
    for t in BAD_THRESHOLDS:
        rows.append((f"bad > {t}m", f"bad_{t}", "{:.1%}"))

    label_w = max(len(label) for label, _, _ in rows)
    col_w = max(12, *(len(t) for t in tags))

    header = "  " + " " * label_w + "  " + "  ".join(f"{t:>{col_w}}" for t in tags)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, key, fmt in rows:
        cells = []
        for t in tags:
            value = results[t].get(key)
            cells.append(fmt.format(value) if value is not None else "-")
        print("  " + f"{label:<{label_w}}" + "  " + "  ".join(f"{c:>{col_w}}" for c in cells))


def render_error_maps(pred_maps: dict[str, np.ndarray], gt: np.ndarray, out_dir: Path) -> None:
    """Save one absolute-error map per tag (white = invalid), sharing a single color
    scale with a 99th-percentile vmax so the backends are visually comparable."""

    error_maps: dict[str, np.ndarray] = {}

    for tag, pred in pred_maps.items():
        gt_eval = align_gt_to_prediction(gt, pred.shape, tag)

        both = np.isfinite(pred) & np.isfinite(gt_eval)

        err = np.full(pred.shape, np.nan, dtype=np.float32)
        err[both] = np.abs(pred[both] - gt_eval[both])
        error_maps[tag] = err

    finite_parts = [e[np.isfinite(e)].ravel() for e in error_maps.values()]

    finite_parts = [x for x in finite_parts if x.size > 0]

    if not finite_parts:
        print("  no overlapping pixels to render error maps")
        return

    finite = np.concatenate(finite_parts)
    vmax = float(np.percentile(finite, 99))

    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("white")  # invalid (NaN) pixels -> white

    for tag, err in error_maps.items():
        fig, ax = plt.subplots(figsize=(9, 6))
        im = ax.imshow(err, cmap=cmap, vmin=0.0, vmax=vmax, interpolation="nearest")
        ax.set_title(f"Absolute depth error — {tag} (white = invalid)")
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("|error| (m)")
        fig.tight_layout()
        dest = out_dir / f"error_{tag}_scaled.png"
        fig.savefig(dest, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {dest.name}")
        


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
