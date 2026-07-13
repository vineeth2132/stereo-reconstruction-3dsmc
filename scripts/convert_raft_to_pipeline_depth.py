from pathlib import Path

import cv2
import numpy as np
import tifffile


OUT_DIR = Path("out")
RAFT_DISP_PATH = Path("out_dl/disparity_raft_small.npy")

RECTIFIED_LEFT_PATH = OUT_DIR / "rectified_left.png"

DEPTH_OUT = OUT_DIR / "depth_raft_float.tiff"
MASK_OUT = OUT_DIR / "valid_depth_raft_mask.tiff"
DISP_OUT = OUT_DIR / "disparity_raft_float.tiff"


FX_FULL = 3408.59
CAMERA_WIDTH_FULL = 6208
BASELINE = 0.4289


def main():
    if not RAFT_DISP_PATH.exists():
        raise FileNotFoundError(f"Missing RAFT disparity: {RAFT_DISP_PATH}")

    disparity = np.load(str(RAFT_DISP_PATH)).astype(np.float32)
    disparity = np.squeeze(disparity)
    disparity = np.abs(disparity)

    h, w = disparity.shape

    fx_scaled = FX_FULL * (w / CAMERA_WIDTH_FULL)

    valid = np.isfinite(disparity) & (disparity > 1e-6)

    depth = np.full((h, w), -1.0, dtype=np.float32)
    depth[valid] = (fx_scaled * BASELINE) / disparity[valid]

    valid_depth = valid & np.isfinite(depth) & (depth > 0.0)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[valid_depth] = 255

    disparity_out = np.full((h, w), -1.0, dtype=np.float32)
    disparity_out[valid] = disparity[valid]

    tifffile.imwrite(str(DEPTH_OUT), depth.astype(np.float32))
    tifffile.imwrite(str(MASK_OUT), mask)
    tifffile.imwrite(str(DISP_OUT), disparity_out.astype(np.float32))

    print("RAFT converted to pipeline format")
    print("---------------------------------")
    print(f"Input disparity: {RAFT_DISP_PATH}")
    print(f"Resolution:      {w} x {h}")
    print(f"Scaled fx:       {fx_scaled:.3f}")
    print(f"Baseline:        {BASELINE:.4f} m")
    print(f"Wrote:           {DEPTH_OUT}")
    print(f"Wrote:           {MASK_OUT}")
    print(f"Wrote:           {DISP_OUT}")


if __name__ == "__main__":
    main()