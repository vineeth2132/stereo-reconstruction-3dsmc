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
import matplotlib
matplotlib.use("Agg")  # file output only, no display
import matplotlib.pyplot as plt


# Bad-pixel error thresholds in meters (depth domain).
BAD_THRESHOLDS = (0.1, 0.5, 1.0)

# Bad-pixel error thresholds in full-res disparity pixels. These mirror the metric
# the RAFT-Stereo comparison reports, computed against a GT disparity synthesized
# from the GT depth (see disparity_metrics).
BAD_DISP_THRESHOLDS = (1.0, 2.0, 4.0)

# Pre-fill failure-reason codes written by the custom matcher into reason_custom.png
# (see CustomMatchReason in include/DenseStereoMatcher.h).
REASON_LABELS = {
    0: "matched",
    1: "never matched",
    2: "border / out-of-image",
    3: "LR-rejected",
    4: "speckle-removed",
}


def load_reason(out_dir: Path) -> np.ndarray | None:
    """Load the raw indexed 0..4 failure-reason PNG as a uint8 array. Returns None if
    the file is missing. Prefers imageio (numpy-native, lossless), then Pillow, and
    only falls back to matplotlib.pyplot.imread (which returns floats in [0, 1] for a
    PNG, so the labels are rescaled back to 0..4)."""
    path = out_dir / "reason_custom.png"
    if not path.exists():
        return None

    arr: np.ndarray
    try:
        import imageio.v2 as imageio  # type: ignore
        arr = np.asarray(imageio.imread(path))
    except Exception:
        try:
            from PIL import Image  # type: ignore
            arr = np.asarray(Image.open(path))
        except Exception:
            arr = plt.imread(path)
            if arr.dtype != np.uint8:
                arr = np.rint(arr * 255.0).astype(np.uint8)

    if arr.ndim == 3:
        arr = arr[..., 0]  # single-channel labels, drop any color replication
    return arr.astype(np.uint8)


def load_matched_mask(out_dir: Path) -> np.ndarray | None:
    """Load the pre-fill matched mask (valid_matched_custom_mask.tiff) as a boolean
    array. Returns None if the file is missing."""
    path = out_dir / "valid_matched_custom_mask.tiff"
    if not path.exists():
        return None
    return tifffile.imread(path) > 0


def load_disparity(out_dir: Path, tag: str) -> np.ndarray | None:
    """Load a full-res disparity map (px) and its mask, returning a float array with
    NaN at invalid pixels. Returns None if the value file is missing."""
    value_path = out_dir / f"disparity_{tag}_float.tiff"
    mask_path = out_dir / f"valid_disparity_{tag}_mask.tiff"
    if not value_path.exists():
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


def disparity_metrics(disp: np.ndarray, depth: np.ndarray, gt: np.ndarray) -> dict:
    """Disparity-domain bad-pixel ratios. Synthesizes a GT disparity from the GT depth
    via the constant fB = focal*baseline, recovered as the median of disp*depth over
    the tag's own jointly-valid pixels (the same derivation scripts/visualize_maps.py
    uses). Compares our disparity to gt_disparity = fB / gt_depth over pixels valid in
    both, at 1 / 2 / 4 px."""
    own = np.isfinite(disp) & np.isfinite(depth)
    if not own.any():
        return {}

    fB = float(np.median(disp[own].astype(np.float64) * depth[own].astype(np.float64)))
    gt_disp = np.full(gt.shape, np.nan, dtype=np.float64)
    gt_valid = np.isfinite(gt)
    gt_disp[gt_valid] = fB / gt[gt_valid]

    both = np.isfinite(disp) & np.isfinite(gt_disp)
    metrics: dict = {"fB": fB}
    if both.any():
        abs_err = np.abs(disp[both].astype(np.float64) - gt_disp[both])
        for t in BAD_DISP_THRESHOLDS:
            metrics[f"bad_disp_{t}"] = float((abs_err > t).mean())
    return metrics


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


def evaluate(pred: np.ndarray, gt: np.ndarray) -> dict | None:
    """Compute error metrics over the pixels valid in both maps. Returns None if
    there is no overlap."""
    both = np.isfinite(pred) & np.isfinite(gt)
    count = int(both.sum())
    gt_valid = int(np.isfinite(gt).sum())
    if count == 0:
        return None

    p = pred[both].astype(np.float64)
    g = gt[both].astype(np.float64)
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
    for t in BAD_DISP_THRESHOLDS:
        rows.append((f"bad > {int(t)}px", f"bad_disp_{t}", "{:.1%}"))

    label_w = max(len(label) for label, _, _ in rows)
    col_w = max(14, *(len(t) for t in tags))

    header = "  " + " " * label_w + "  " + "  ".join(f"{t:>{col_w}}" for t in tags)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, key, fmt in rows:
        cells = []
        for t in tags:
            value = results[t].get(key)
            cells.append(fmt.format(value) if value is not None else "-")
        print("  " + f"{label:<{label_w}}" + "  " + "  ".join(f"{c:>{col_w}}" for c in cells))


def print_reason_breakdown(reason: np.ndarray, gt_valid: np.ndarray,
                           custom_depth: np.ndarray | None) -> None:
    """Report the pre-fill failure-reason distribution over GT-valid pixels, plus how
    much of the final custom depth (over GT-valid pixels) came from hole filling."""
    if reason.shape != gt_valid.shape:
        print(f"  reason map {reason.shape} does not match GT {gt_valid.shape}; skipping breakdown")
        return

    total = int(gt_valid.sum())
    if total == 0:
        return

    print("\nPre-fill failure reasons over GT-valid pixels (custom):")
    reason_gt = reason[gt_valid]
    for code, label in REASON_LABELS.items():
        pct = 100.0 * int(np.count_nonzero(reason_gt == code)) / total
        print(f"  {label:<24} {pct:6.2f}%")

    if custom_depth is not None:
        final_valid = np.isfinite(custom_depth) & gt_valid
        n = int(final_valid.sum())
        if n:
            filled = int(np.count_nonzero(final_valid & (reason != 0)))
            print(f"  {'filled (of final-valid)':<24} {100.0 * filled / n:6.2f}%")


def render_error_maps(pred_maps: dict[str, np.ndarray], gt: np.ndarray, out_dir: Path) -> None:
    """Save one absolute-error map per tag (white = invalid), sharing a single color
    scale with a 99th-percentile vmax so the backends are visually comparable."""
    error_maps: dict[str, np.ndarray] = {}
    for tag, pred in pred_maps.items():
        both = np.isfinite(pred) & np.isfinite(gt)
        err = np.full(pred.shape, np.nan, dtype=np.float32)
        err[both] = np.abs(pred[both] - gt[both])
        error_maps[tag] = err

    finite = np.concatenate([e[np.isfinite(e)].ravel() for e in error_maps.values()])
    if finite.size == 0:
        print("  no overlapping pixels to render error maps")
        return
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
    disp_maps: dict[str, np.ndarray] = {}
    results: dict[str, dict] = {}
    for tag in args.tags:
        pred = load_pred(out_dir, tag)
        if pred is None:
            continue
        pred_maps[tag] = pred
        metrics = evaluate(pred, gt)
        if metrics is None:
            print(f"  skip {tag}: no overlap with GT")
            continue

        disp = load_disparity(out_dir, tag)
        if disp is not None:
            disp_maps[tag] = disp
            metrics.update(disparity_metrics(disp, pred, gt))
        results[tag] = metrics

    # Honest, fill-free "custom-matched" row: the custom maps restricted to the
    # genuinely matched pixels (pre-fill), in both the depth and disparity domains.
    matched_mask = load_matched_mask(out_dir)
    if "custom" in pred_maps and matched_mask is not None and matched_mask.shape == gt.shape:
        pred_m = pred_maps["custom"].copy()
        pred_m[~matched_mask] = np.nan
        metrics_m = evaluate(pred_m, gt)
        if metrics_m is not None:
            if "custom" in disp_maps:
                disp_m = disp_maps["custom"].copy()
                disp_m[~matched_mask] = np.nan
                metrics_m.update(disparity_metrics(disp_m, pred_m, gt))
            results["custom-matched"] = metrics_m

    if not results:
        raise SystemExit("No backends could be evaluated (no overlap with GT).")

    print("\nDepth evaluation vs. rectified ETH3D ground truth:")
    print_table(results)

    reason = load_reason(out_dir)
    if reason is not None:
        print_reason_breakdown(reason, np.isfinite(gt), pred_maps.get("custom"))

    print("\nError maps:")
    render_error_maps(pred_maps, gt, out_dir)


if __name__ == "__main__":
    main()
