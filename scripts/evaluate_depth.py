#!/usr/bin/env python3
"""Quantitative depth evaluation against the rectified ETH3D ground truth.

The C++ pipeline writes one metric depth map per dense-matching backend/stage:

    out/depth_<tag>_float.tiff   + out/valid_depth_<tag>_mask.tiff   (invalid = -1)

and, where available, the matching disparity map:

    out/disparity_<tag>_float.tiff + out/valid_disparity_<tag>_mask.tiff

and one rectified ground-truth depth map (metric meters, invalid = NaN):

    out/gt_depth_rectified_float.tiff   (full resolution, e.g. 6208x4135)

Evaluation convention (TA-prescribed): every method runs at its own native
resolution and its prediction is RESAMPLED UP to the full-resolution GT frame,
where all metrics and error maps are computed. This scores every method in the
same full-res GT frame, which is the fair comparison. Upsampling uses nearest
neighbour so NaN-invalid pixels never smear into valid ones.

Full-resolution backends (same size as the GT) are passed through unchanged.
Deep methods run smaller and are mapped up:

  * RAFT-Stereo (~640x416): a plain cv2.resize back to the GT size inverts its
    aspect-preserving downscale. Its disparity is in RAFT-input pixels, so the
    disparity values are multiplied by the width scale (gt_w / native_w) to put
    them into full-res pixel units.
  * DUSt3R (512x336): the long edge (width) was scaled to 512 and the image was
    centre-cropped vertically. We invert that: upscale by gt_w/512 and paste the
    block into a NaN canvas at the vertical crop offset. Rows DUSt3R never saw
    stay NaN (counted as not covered). DUSt3R has no disparity file.

For each tag we intersect our-valid with GT-valid and report coverage, MAE,
RMSE, median absolute error, depth bad-pixel ratios, a median-scale-aligned
MAE (removes a global scale bias), and disparity bad-pixel ratios at 1/2/4 px
where a disparity map exists. A compact comparison table is printed and
absolute-error maps (white = invalid, shared color scale, in the GT frame) are
saved as out/error_<tag>_scaled.png.

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


# Bad-pixel depth error thresholds in meters.
BAD_THRESHOLDS = (0.1, 0.5, 1.0)

# Bad-pixel disparity error thresholds in pixels (full-res GT frame).
BAD_DISP_THRESHOLDS = (1.0, 2.0, 4.0)

# DUSt3R input geometry, matching scripts/run_dust3r_depth.py (load_images).
DUST3R_LONG_EDGE = 512
DUST3R_PATCH_SIZE = 16

DEFAULT_TAGS = [
    "opencv_final",
    "NCC_raw", "NCC_filtered", "NCC_filled",
    "SSD_raw", "SSD_filtered", "SSD_filled",
    "Census_raw", "Census_filtered", "Census_filled",
    "dust3r", "raft",
]


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


def load_disparity(out_dir: Path, tag: str) -> np.ndarray | None:
    """Load a disparity map with NaN at invalid pixels, or None if absent.

    The OpenCV disparity files are named with the bare ``opencv`` tag even
    though its depth uses ``opencv_final``; that mapping is handled here.
    """
    disp_tag = "opencv" if tag == "opencv_final" else tag
    value_path = out_dir / f"disparity_{disp_tag}_float.tiff"
    mask_path = out_dir / f"valid_disparity_{disp_tag}_mask.tiff"

    if not value_path.exists():
        return None

    values = tifffile.imread(value_path).astype(np.float32)
    if mask_path.exists():
        mask = tifffile.imread(mask_path) > 0
    else:
        # RAFT stores invalid disparity as -1 with no separate mask.
        mask = np.isfinite(values) & (values > 0.0)

    out = values.copy()
    out[~mask] = np.nan
    out[~np.isfinite(out)] = np.nan
    out[out <= 0.0] = np.nan
    return out


def _dust3r_crop_top(gt_h: int, gt_w: int) -> int:
    """Vertical crop offset (in the long-edge-512 resized frame) that DUSt3R's
    load_images applies. Replicates the exact centred patch crop math."""
    size = DUST3R_LONG_EDGE
    patch = DUST3R_PATCH_SIZE

    long_edge = max(gt_w, gt_h)
    resized_h = int(round(gt_h * size / long_edge))
    cy = resized_h // 2
    half_h = ((2 * cy) // patch) * patch / 2.0
    return int(cy - half_h)


def align_prediction_to_gt(
    pred: np.ndarray,
    gt_shape: tuple[int, int],
    tag: str,
) -> np.ndarray:
    """Upsample a prediction to the full-resolution GT frame.

    Nearest-neighbour keeps NaN-invalid pixels clean. Full-resolution
    predictions (shape already equal to the GT) are returned unchanged so
    their metrics stay bit-identical to a direct comparison.
    """
    gt_h, gt_w = gt_shape

    if pred.shape == gt_shape:
        return pred.copy()

    native_h, native_w = pred.shape

    if "dust3r" in tag:
        # Invert long-edge-512 resize + vertical centre crop.
        scale = gt_w / native_w
        up_h = int(round(native_h * scale))

        resized = cv2.resize(
            pred.astype(np.float32),
            (gt_w, up_h),
            interpolation=cv2.INTER_NEAREST,
        )

        crop_top = _dust3r_crop_top(gt_h, gt_w)
        offset = int(round(crop_top * scale))

        canvas = np.full(gt_shape, np.nan, dtype=np.float32)
        end = min(offset + up_h, gt_h)
        if offset < gt_h and end > offset:
            canvas[offset:end, :] = resized[: end - offset, :]
        return canvas

    # RAFT and any other lower-resolution prediction: direct resize.
    return cv2.resize(
        pred.astype(np.float32),
        (gt_w, gt_h),
        interpolation=cv2.INTER_NEAREST,
    )


def align_disparity_to_gt(
    disp: np.ndarray,
    gt_shape: tuple[int, int],
    tag: str,
) -> np.ndarray:
    """Upsample a disparity map to the GT frame, scaling the disparity VALUES by
    the width scale (gt_w / native_w) so they are in full-res pixel units."""
    gt_h, gt_w = gt_shape

    if disp.shape == gt_shape:
        return disp.copy()

    native_h, native_w = disp.shape
    width_scale = gt_w / native_w

    up = cv2.resize(
        disp.astype(np.float32),
        (gt_w, gt_h),
        interpolation=cv2.INTER_NEAREST,
    )
    return up * np.float32(width_scale)


def evaluate(pred: np.ndarray, gt: np.ndarray) -> dict | None:
    """Compute direct and median-scale-aligned depth metrics.

    Both arrays are already in the full-resolution GT frame.

    Direct metrics use the predicted metric depth unchanged. Scaled metrics
    first multiply the prediction by the median GT/prediction depth ratio,
    removing one global scale bias while preserving the predicted structure.
    """
    both = np.isfinite(pred) & np.isfinite(gt)
    count = int(both.sum())
    gt_valid = int(np.isfinite(gt).sum())

    if count == 0:
        return None

    p = pred[both].astype(np.float64)
    g = gt[both].astype(np.float64)

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
        metrics[f"bad_{threshold}"] = float(np.mean(abs_err > threshold))
        metrics[f"bad_scaled_{threshold}"] = float(
            np.mean(abs_err_scaled > threshold)
        )

    return metrics


def add_disparity_metrics(
    metrics: dict,
    disp: np.ndarray,
    depth: np.ndarray,
    gt: np.ndarray,
) -> None:
    """Add disparity bad-pixel ratios (1/2/4 px) in the full-res GT frame.

    f*B is estimated as median(disp * depth) over the method's own valid pixels,
    a GT-disparity map is built as gt_disp = f*B / gt, and the fraction of pixels
    with |disp - gt_disp| above each threshold is reported. All inputs are in the
    GT frame; for upsampled low-res disparity the values were already scaled to
    full-res pixel units so f*B stays consistent with the unchanged depth.
    """
    own = np.isfinite(disp) & np.isfinite(depth)
    if not own.any():
        return

    f_b = float(np.median(disp[own].astype(np.float64) * depth[own].astype(np.float64)))
    metrics["fB"] = f_b

    gt_disp = np.full(gt.shape, np.nan, dtype=np.float64)
    gt_valid = np.isfinite(gt)
    gt_disp[gt_valid] = f_b / gt[gt_valid]

    both = np.isfinite(disp) & np.isfinite(gt_disp)
    if not both.any():
        return

    abs_err = np.abs(disp[both].astype(np.float64) - gt_disp[both])
    for threshold in BAD_DISP_THRESHOLDS:
        metrics[f"bad_disp_{threshold}"] = float(np.mean(abs_err > threshold))


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
        rows.append((f"bad > {threshold}m", f"bad_{threshold}", "{:.1%}"))
        rows.append(
            (f"bad scaled > {threshold}m", f"bad_scaled_{threshold}", "{:.1%}")
        )

    for threshold in BAD_DISP_THRESHOLDS:
        rows.append((f"bad > {int(threshold)}px", f"bad_disp_{threshold}", "{:.1%}"))

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
    """Save median-scale-aligned absolute-error maps in the full-res GT frame.

    Every prediction is already aligned to the GT resolution. White pixels are
    invalid. All backends share one color scale.
    """
    error_maps: dict[str, np.ndarray] = {}

    for tag, pred in pred_maps.items():
        both = np.isfinite(pred) & np.isfinite(gt)

        err_map = np.full(pred.shape, np.nan, dtype=np.float32)

        if not np.any(both):
            error_maps[tag] = err_map
            continue

        p = pred[both].astype(np.float64)
        g = gt[both].astype(np.float64)

        valid_ratio = np.isfinite(g / p) & (p > 0.0)

        if not np.any(valid_ratio):
            error_maps[tag] = err_map
            continue

        scale = float(np.median(g[valid_ratio] / p[valid_ratio]))

        err_map[both] = np.abs(
            pred[both].astype(np.float32) * np.float32(scale)
            - gt[both].astype(np.float32)
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
            f"Scale-aligned absolute depth error — {tag} (white = invalid)"
        )
        ax.set_xticks([])
        ax.set_yticks([])

        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label("|scaled error| (m)")

        fig.tight_layout()

        destination = out_dir / f"error_{tag}_scaled.png"
        fig.savefig(destination, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"  wrote {destination.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent / "out"),
        help="Directory holding the .tiff maps (default: ../out)",
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        default=DEFAULT_TAGS,
        help="Backend tags to evaluate (default: all classical + dust3r + raft)",
    )
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
    gt_h, gt_w = gt.shape
    print(
        f"Loaded GT: {out_dir / 'gt_depth_rectified_float.tiff'} "
        f"({gt_w}x{gt_h}, {int(np.isfinite(gt).sum()):,} valid px)"
    )

    pred_maps: dict[str, np.ndarray] = {}
    results: dict[str, dict] = {}
    for tag in args.tags:
        pred = load_pred(out_dir, tag)
        if pred is None:
            continue

        native_h, native_w = pred.shape
        print(f"  {tag}: native {native_w}x{native_h}, evaluated at {gt_w}x{gt_h}")

        pred_gt = align_prediction_to_gt(pred, gt.shape, tag)

        metrics = evaluate(pred_gt, gt)
        if metrics is None:
            print(f"  skip {tag}: no overlap with GT")
            continue

        disp = load_disparity(out_dir, tag)
        if disp is not None:
            disp_gt = align_disparity_to_gt(disp, gt.shape, tag)
            add_disparity_metrics(metrics, disp_gt, pred_gt, gt)

        pred_maps[tag] = pred_gt
        results[tag] = metrics

    if not results:
        raise SystemExit("No backends could be evaluated (no overlap with GT).")

    print("\nDepth evaluation vs. rectified ETH3D ground truth:")
    print_table(results)

    print("\nError maps:")
    render_error_maps(pred_maps, gt, out_dir)


if __name__ == "__main__":
    main()
