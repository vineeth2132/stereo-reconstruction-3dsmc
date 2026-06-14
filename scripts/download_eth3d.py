"""Download and extract an ETH3D high-resolution multi-view scene.

ETH3D scenes are non-rectified (TA requirement). The 'undistorted' archive
contains the DSLR images plus COLMAP-format calibration (cameras.txt with
intrinsics, images.txt with per-image poses -> camera baseline). The 'depth'
archive contains ground-truth depth maps used later by the evaluation harness.

Usage (from the repo root, using the project venv):
    .venv\\Scripts\\python.exe scripts\\download_eth3d.py                 # images + calib
    .venv\\Scripts\\python.exe scripts\\download_eth3d.py --with-depth    # also GT depth
    .venv\\Scripts\\python.exe scripts\\download_eth3d.py --scene electro

Requires: py7zr  (pip install py7zr)  -- already in the project venv.
"""

import argparse
import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://www.eth3d.net/data"

# archive suffixes that exist per scene
ARCHIVES = {
    "images": "{scene}_dslr_undistorted.7z",   # images + dslr_calibration_undistorted/
    "depth": "{scene}_dslr_depth.7z",          # ground-truth depth maps
    "occlusion": "{scene}_dslr_occlusion.7z",  # occlusion masks (for fair eval)
}


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[skip] already downloaded: {dest.name}")
        return
    print(f"[get ] {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    def _progress(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(100.0, block_num * block_size * 100.0 / total_size)
            sys.stdout.write(f"\r       {pct:5.1f}%")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, tmp, _progress)
    sys.stdout.write("\n")
    tmp.rename(dest)


def extract(archive: Path, dest_dir: Path) -> None:
    import py7zr
    print(f"[7z  ] extracting {archive.name} -> {dest_dir}")
    with py7zr.SevenZipFile(archive, mode="r") as z:
        z.extractall(path=dest_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download an ETH3D multi-view scene.")
    parser.add_argument("--scene", default="delivery_area", help="scene name (default: delivery_area)")
    parser.add_argument("--dest", default="data", help="destination data directory (default: data)")
    parser.add_argument("--with-depth", action="store_true", help="also download ground-truth depth maps")
    parser.add_argument("--with-occlusion", action="store_true", help="also download occlusion masks")
    parser.add_argument("--keep-archives", action="store_true", help="do not delete .7z files after extraction")
    args = parser.parse_args()

    dest_dir = Path(args.dest)
    parts = ["images"]
    if args.with_depth:
        parts.append("depth")
    if args.with_occlusion:
        parts.append("occlusion")

    for part in parts:
        name = ARCHIVES[part].format(scene=args.scene)
        url = f"{BASE_URL}/{name}"
        archive = dest_dir / name
        download(url, archive)
        extract(archive, dest_dir)
        if not args.keep_archives:
            archive.unlink(missing_ok=True)

    scene_dir = dest_dir / args.scene
    print(f"\nDone. Scene extracted under: {scene_dir.resolve()}")
    print("Expected by the pipeline:")
    print(f"  {scene_dir / 'images' / 'dslr_images_undistorted'} (e.g. DSC_0688.jpg)")
    print(f"  {scene_dir / 'dslr_calibration_undistorted' / 'cameras.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
