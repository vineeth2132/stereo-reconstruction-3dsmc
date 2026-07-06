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


def render_comparison(maps: dict[str, np.ndarray], kind: str, cbar_label: str, dest: Path) -> None:
    """Side-by-side panels sharing one color scale, so the backends are comparable."""
    tags = [t for t in ("opencv", "custom") if t in maps]
    if not tags:
        return

    # Shared scale across panels for a fair visual comparison.
    finite = np.concatenate([maps[t][np.isfinite(maps[t])].ravel() for t in tags])
    vmin = float(np.percentile(finite, 1)) if finite.size else 0.0
    vmax = float(np.percentile(finite, 99)) if finite.size else 1.0

    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("white")

    fig, axes = plt.subplots(1, len(tags), figsize=(9 * len(tags), 6), squeeze=False)
    im = None
    for ax, tag in zip(axes[0], tags):
        im = ax.imshow(maps[tag], cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        valid_pct = 100.0 * np.isfinite(maps[tag]).mean()
        ax.set_title(f"{kind} — {tag}  ({valid_pct:.1f}% valid)")
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

        if len(loaded) >= 2:
            render_comparison(loaded, kind, cbar_label, out_dir / f"comparison_{kind}.png")


if __name__ == "__main__":
    main()
