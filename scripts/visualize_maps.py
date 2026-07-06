#!/usr/bin/env python3
"""Render the disparity / depth maps with a color scale (white = invalid).

The C++ pipeline writes one set of maps per dense-matching backend, tagged
``opencv`` and ``custom``:

    out/disparity_<tag>_float.tiff   + out/valid_disparity_<tag>_mask.tiff
    out/depth_<tag>_float.tiff       + out/valid_depth_<tag>_mask.tiff

This script mirrors the MATLAB helpers (imagesc + colorbar, NaN shown white) and
produces, for the report:

    out/disparity_<tag>_scaled.png   out/depth_<tag>_scaled.png
    out/comparison_disparity.png     out/comparison_depth.png   (opencv vs custom)

Run with the project venv:

    .venv\\Scripts\\python.exe scripts\\visualize_maps.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")  # file output only, no display
import matplotlib.pyplot as plt


def load_map(out_dir: Path, kind: str, tag: str) -> np.ndarray | None:
    """Load a float map and its mask, returning a float array with NaN at invalid
    pixels. ``kind`` is 'disparity' or 'depth'. Returns None if files are missing."""
    if kind == "disparity":
        value_path = out_dir / f"disparity_{tag}_float.tiff"
        mask_path = out_dir / f"valid_disparity_{tag}_mask.tiff"
    else:
        value_path = out_dir / f"depth_{tag}_float.tiff"
        mask_path = out_dir / f"valid_depth_{tag}_mask.tiff"

    if not value_path.exists():
        print(f"  skip {kind}/{tag}: {value_path.name} not found")
        return None

    values = tifffile.imread(value_path).astype(np.float32)
    if mask_path.exists():
        mask = tifffile.imread(mask_path) > 0
    else:
        mask = np.isfinite(values)

    out = values.copy()
    out[~mask] = np.nan
    out[~np.isfinite(out)] = np.nan
    return out


def load_gt_depth(out_dir: Path) -> np.ndarray | None:
    """Load the rectified GT depth (metric meters) as a float array with NaN at
    invalid pixels. Invalid GT is encoded as NaN or non-positive depth (mirrors
    scripts/evaluate_depth.py). Returns None if the file is missing."""
    gt_path = out_dir / "gt_depth_rectified_float.tiff"
    if not gt_path.exists():
        return None

    gt = tifffile.imread(gt_path).astype(np.float32)
    gt[~np.isfinite(gt)] = np.nan
    gt[gt <= 0.0] = np.nan
    return gt


def render_single(data: np.ndarray, title: str, cbar_label: str, dest: Path) -> None:
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("white")  # invalid (NaN) pixels -> white

    finite = data[np.isfinite(data)]
    vmin = float(np.percentile(finite, 1)) if finite.size else 0.0
    vmax = float(np.percentile(finite, 99)) if finite.size else 1.0

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {dest.name}")


def render_comparison(maps: dict[str, np.ndarray], kind: str, cbar_label: str, dest: Path,
                      gt: np.ndarray | None = None,
                      gt_title: str = "Ground truth (laser)") -> None:
    """Side-by-side panels sharing one color scale, so the backends are comparable.
    When ``gt`` is provided it is appended as an extra panel; the shared scale spans
    every panel (backends + GT)."""
    tags = [t for t in ("opencv", "custom") if t in maps]
    if not tags:
        return

    # Ordered panels: backends first, then the ground-truth reference if available.
    panels: list[tuple[str, np.ndarray]] = [(f"{kind} — {tag}", maps[tag]) for tag in tags]
    if gt is not None:
        panels.append((gt_title, gt))

    # Shared scale across ALL panels (backends + GT) for a fair visual comparison.
    finite = np.concatenate([m[np.isfinite(m)].ravel() for _, m in panels])
    vmin = float(np.percentile(finite, 1)) if finite.size else 0.0
    vmax = float(np.percentile(finite, 99)) if finite.size else 1.0

    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("white")

    fig, axes = plt.subplots(1, len(panels), figsize=(9 * len(panels), 6), squeeze=False)
    im = None
    for ax, (title, m) in zip(axes[0], panels):
        im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        # % of the whole image frame that is valid (unambiguous denominator).
        valid_pct = 100.0 * np.isfinite(m).mean()
        ax.set_title(f"{title}  (valid: {valid_pct:.1f}% of image px)")
        ax.set_xticks([])
        ax.set_yticks([])

    cbar = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {dest.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "out"),
                        help="Directory holding the .tiff maps (default: ../out)")
    parser.add_argument("--tags", nargs="+", default=["opencv", "custom"],
                        help="Backend tags to render (default: opencv custom)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    if not out_dir.is_dir():
        raise SystemExit(f"Output directory not found: {out_dir}")

    specs = [
        ("disparity", "Disparity (px)"),
        ("depth", "Depth Z (up to scale)"),
    ]

    gt_depth = load_gt_depth(out_dir)
    if gt_depth is None:
        print("note: gt_depth_rectified_float.tiff not found — "
              "comparison figures fall back to 2 panels (opencv | custom)")

    for kind, cbar_label in specs:
        print(f"{kind}:")
        loaded: dict[str, np.ndarray] = {}
        for tag in args.tags:
            data = load_map(out_dir, kind, tag)
            if data is None:
                continue
            loaded[tag] = data
            title = f"{kind.capitalize()} map — {tag} (white = invalid)"
            render_single(data, title, cbar_label, out_dir / f"{kind}_{tag}_scaled.png")

        # Build the ground-truth panel for this kind (None -> 2-panel fallback).
        gt_panel: np.ndarray | None = None
        if gt_depth is not None:
            if kind == "depth":
                gt_panel = gt_depth
            else:
                # No GT disparity file exists, so derive it. For rectified stereo the
                # product disparity * depth is the constant focal*baseline (fB); recover
                # it as the median over pixels valid in both custom maps, then invert
                # the GT depth: gt_disparity = fB / gt_depth (NaN where GT is invalid).
                disp_custom = loaded.get("custom")
                depth_custom = load_map(out_dir, "depth", "custom")
                if disp_custom is not None and depth_custom is not None:
                    both = np.isfinite(disp_custom) & np.isfinite(depth_custom)
                    if both.any():
                        fB = float(np.median(disp_custom[both] * depth_custom[both]))
                        print(f"  derived fB (focal*baseline) = {fB:.4f}")
                        gt_panel = fB / gt_depth

        if len(loaded) >= 2:
            render_comparison(loaded, kind, cbar_label, out_dir / f"comparison_{kind}.png",
                              gt=gt_panel)


if __name__ == "__main__":
    main()
