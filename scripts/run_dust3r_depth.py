#!/usr/bin/env python3
"""Run DUSt3R on the rectified stereo pair and save metric depth output."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile
import torch


DEFAULT_MODEL = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
DEFAULT_BASELINE_METERS = 0.4289


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dust3r-root",
        type=Path,
        default=project_root.parent / "dust3r",
        help="Path to the cloned DUSt3R repository.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=project_root / "out",
        help="Directory containing rectified images and receiving depth output.",
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
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face DUSt3R model name.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="PyTorch device, normally cuda or cpu.",
    )
    parser.add_argument(
        "--confidence-percentile",
        type=float,
        default=0.0,
        help=(
            "Reject predictions below this confidence percentile. "
            "Use 0 to keep all finite positive depths."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dust3r_root = args.dust3r_root.resolve()
    out_dir = args.out.resolve()

    left_path = out_dir / "rectified_left.png"
    right_path = out_dir / "rectified_right.png"

    for path in (dust3r_root, left_path, right_path):
        if not path.exists():
            raise SystemExit(f"Required path not found: {path}")

    if args.baseline is None:
        # The baseline changes per scene/pair, so prefer the value the C++
        # pipeline recorded next to the rectified images.
        meta_path = out_dir / "rectified_meta.txt"
        if meta_path.exists():
            for line in meta_path.read_text().splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[0] == "baseline":
                    args.baseline = float(parts[1])
                    print(f"Using baseline from rectified_meta.txt: {args.baseline:.6f} m")
                    break
        if args.baseline is None:
            args.baseline = DEFAULT_BASELINE_METERS
            print(
                "Warning: rectified_meta.txt not found; falling back to the "
                f"delivery_area default baseline = {DEFAULT_BASELINE_METERS} m."
            )

    if args.baseline <= 0:
        raise SystemExit("--baseline must be positive")

    if not 0.0 <= args.confidence_percentile < 100.0:
        raise SystemExit("--confidence-percentile must be in [0, 100)")

    sys.path.insert(0, str(dust3r_root))

    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.model import AsymmetricCroCo3DStereo
    from dust3r.utils.image import load_images

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is False")

    print(f"Loading DUSt3R model: {args.model}")
    model = AsymmetricCroCo3DStereo.from_pretrained(args.model).to(args.device)
    model.eval()

    square_ok = getattr(model, "square_ok", False)

    print("Loading rectified stereo images...")
    images = load_images(
        [str(left_path), str(right_path)],
        size=512,
        patch_size=model.patch_size,
        square_ok=square_ok,
    )

    if len(images) != 2:
        raise SystemExit(f"Expected 2 images, received {len(images)}")

    pairs = make_pairs(
        images,
        scene_graph="complete",
        prefilter=None,
        symmetrize=True,
    )

    print("Running DUSt3R inference...")
    output = inference(
        pairs,
        model,
        device=args.device,
        batch_size=1,
        verbose=True,
    )

    print("Recovering the two-view scene with PairViewer...")
    scene = global_aligner(
        output,
        device=args.device,
        mode=GlobalAlignerMode.PairViewer,
        verbose=True,
    )

    poses = scene.get_im_poses().detach().cpu().numpy()
    depthmaps = scene.get_depthmaps()
    confidences = scene.get_conf()

    if poses.shape != (2, 4, 4):
        raise SystemExit(f"Unexpected camera pose shape: {poses.shape}")

    predicted_baseline = float(
        np.linalg.norm(poses[1, :3, 3] - poses[0, :3, 3])
    )

    if not np.isfinite(predicted_baseline) or predicted_baseline <= 1e-8:
        raise SystemExit(
            f"DUSt3R returned an invalid relative baseline: {predicted_baseline}"
        )

    metric_scale = args.baseline / predicted_baseline

    relative_depth = depthmaps[0].detach().cpu().numpy().astype(np.float32)
    confidence = confidences[0].detach().cpu().numpy().astype(np.float32)

    if relative_depth.shape != confidence.shape:
        raise SystemExit(
            "Depth and confidence shapes differ: "
            f"{relative_depth.shape} vs {confidence.shape}"
        )

    metric_depth = relative_depth * np.float32(metric_scale)

    valid = (
        np.isfinite(metric_depth)
        & (metric_depth > 0.0)
        & np.isfinite(confidence)
    )

    if args.confidence_percentile > 0.0:
        finite_confidence = confidence[valid]
        if finite_confidence.size == 0:
            raise SystemExit("No valid confidence values were produced")

        confidence_threshold = float(
            np.percentile(finite_confidence, args.confidence_percentile)
        )
        valid &= confidence >= confidence_threshold
    else:
        confidence_threshold = None

    if not valid.any():
        raise SystemExit("DUSt3R produced an empty valid-depth mask")

    depth_output = np.full(metric_depth.shape, -1.0, dtype=np.float32)
    depth_output[valid] = metric_depth[valid]

    mask_output = (valid.astype(np.uint8) * 255)

    out_dir.mkdir(parents=True, exist_ok=True)

    depth_path = out_dir / "depth_dust3r_float.tiff"
    mask_path = out_dir / "valid_depth_dust3r_mask.tiff"

    tifffile.imwrite(depth_path, depth_output)
    tifffile.imwrite(mask_path, mask_output)

    valid_depth = depth_output[valid]

    print()
    print("DUSt3R depth completed")
    print(f"  input resolution: 6208x4135")
    print(
        "  DUSt3R resolution: "
        f"{metric_depth.shape[1]}x{metric_depth.shape[0]}"
    )
    print(f"  physical baseline: {args.baseline:.6f} m")
    print(f"  predicted relative baseline: {predicted_baseline:.8f}")
    print(f"  metric scale: {metric_scale:.6f}")
    print(f"  valid pixels: {int(valid.sum()):,} / {valid.size:,}")

    if confidence_threshold is not None:
        print(f"  confidence threshold: {confidence_threshold:.6f}")

    print(
        "  metric depth range: "
        f"{float(valid_depth.min()):.4f} to "
        f"{float(valid_depth.max()):.4f} m"
    )
    print(f"  median metric depth: {float(np.median(valid_depth)):.4f} m")
    print(f"  wrote: {depth_path}")
    print(f"  wrote: {mask_path}")


if __name__ == "__main__":
    main()