# Stereo Reconstruction

A C++ stereo reconstruction pipeline implemented with OpenCV and Eigen.

The current pipeline performs:

1. Stereo image loading
2. Camera intrinsics loading
3. SIFT keypoint detection
4. Sparse feature matching with ratio filtering
5. Fundamental matrix estimation using a custom normalized 8-point algorithm with custom RANSAC, with OpenCV kept only for comparison/debugging
6. Essential matrix computation
7. Relative camera pose recovery
8. Stereo rectification
9. Dense disparity-map generation with StereoBM or StereoSGBM, optionally refined
   with an edge-aware WLS filter (`opencv_contrib` ximgproc) and a median + WLS
   confidence post-filter
10. Depth reconstruction: disparity → 3D points via the rectification `Q` matrix
11. Colored point-cloud export (`.ply`)
12. Triangle-mesh export (`.ply`) from the disparity grid, with depth-discontinuity culling

Each stage also writes a step output to the `out/` directory (see [Outputs](#outputs))
so the pipeline can be inspected without the blocking `imshow` debug windows.

## Requirements

* CMake 3.20 or newer
* A C++20-compatible compiler
* [vcpkg](https://github.com/microsoft/vcpkg)
* OpenCV
* Eigen3

The required libraries are listed in `vcpkg.json`:

```json
{
  "dependencies": [
    "opencv4",
    "eigen3"
  ]
}
```

The WLS disparity filter additionally needs the `ximgproc` module from
`opencv_contrib`. Install OpenCV with the `contrib` feature to enable it:

```powershell
vcpkg install "opencv4[contrib]:x64-windows" --recurse
```

This is optional — without it the project still builds, falling back to
unfiltered SGBM (CMake reports which path is active). The `useWlsFilter` flag in
`DenseStereoConfig` is ignored when the module is absent.

## Project Structure

```text
stereo-reconstruction/
├── CMakeLists.txt
├── vcpkg.json
├── include/
├── src/
├── scripts/      # dataset download helper (download_eth3d.py)
├── data/         # input images + calibration (images are git-ignored)
├── out/          # generated step outputs (git-ignored)
└── build/        # generated build files (git-ignored)
```

The `build/` directory contains generated build files and should not be committed.

## Dataset

The current implementation has been tested with the **ETH3D `delivery_area` (DSLR, undistorted) dataset**. ETH3D is non-rectified, which is required for this project.

The image files are **not committed** (they are git-ignored); only `data/delivery_area/dslr_calibration_undistorted/cameras.txt` is tracked. Download the images with the helper script (uses the project venv, no 7-Zip needed):

```powershell
.venv\Scripts\python.exe scripts\download_eth3d.py              # images + calibration (~0.5 GB)
.venv\Scripts\python.exe scripts\download_eth3d.py --with-depth # also ground-truth depth (for evaluation)
```

This extracts to `data/delivery_area/`, matching the paths in `src/main.cpp`. Update those paths if your local dataset structure differs.

Example:

```cpp
DataLoader dataLoader(
    "../data/delivery_area/images/dslr_images_undistorted/");
```

The camera intrinsics file should follow the COLMAP `cameras.txt` format.

Example:

```text
0 PINHOLE 6208 4135 3408.59 3408.87 3117.24 2064.07
```


## Build Instructions

Replace `<VCPKG_ROOT>` with the path to your local vcpkg installation.

### Windows

Open PowerShell in the project root directory:

```powershell
mkdir build
cd build

cmake .. `
  -DCMAKE_TOOLCHAIN_FILE="<VCPKG_ROOT>\scripts\buildsystems\vcpkg.cmake"

cmake --build . --config Release
```

Run:

```powershell
.\Release\stereo_reconstruction.exe
```

For a debug build:

```powershell
cmake --build . --config Debug
.\Debug\stereo_reconstruction.exe
```

The configure step also generates a Visual Studio solution inside `build/`. It can be opened directly:

```text
build/stereo_reconstruction.sln
```

### Linux

Open a terminal in the project root directory:

```bash
mkdir build
cd build

cmake .. \
  -DCMAKE_TOOLCHAIN_FILE="<VCPKG_ROOT>/scripts/buildsystems/vcpkg.cmake"

cmake --build . --parallel
```

Run:

```bash
./stereo_reconstruction
```

## Reconfigure After Adding Source Files

When a new `.cpp` or `.h` file is added, update `CMakeLists.txt`.

Then regenerate the build files:

```bash
cd build
cmake ..
```

## Clean Build

If the build configuration becomes inconsistent, delete the generated build directory and configure again.

### Windows

```powershell
Remove-Item -Recurse -Force build
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE="<VCPKG_ROOT>\scripts\buildsystems\vcpkg.cmake"
```

### Linux

```bash
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE="<VCPKG_ROOT>/scripts/buildsystems/vcpkg.cmake"
```

## Outputs

Running the executable writes the following step outputs to `out/`:

| File | Description |
|------|-------------|
| `rectified_left.png`, `rectified_right.png` | Rectified pair — corresponding points should lie on the same rows |
| `disparity_colored.png` | Colored disparity map (invalid pixels black) |
| `disparity_float.tiff`, `valid_disparity_mask.tiff` | Raw disparity + mask (for MATLAB/quantitative inspection) |
| `depth_float.tiff`, `valid_depth_mask.tiff` | Per-pixel depth (Z) + mask |
| `pointcloud.ply` | Colored 3D point cloud |
| `mesh.ply` | Triangle mesh from the disparity grid |

The `scripts/visualize_disparity.m` / `visualize_depth.m` MATLAB helpers display
the `.tiff` maps with a color scale.

Open the `.ply` files in MeshLab or CloudCompare. `out/` is git-ignored.

## Depth reconstruction (`DepthReconstructor`)

`DepthReconstructor` turns the disparity map into 3D geometry:

* Uses the rectification `Q` matrix (`cv::reprojectImageTo3D`) to back-project each
  valid disparity pixel into a 3D point, colored from the rectified left image.
* The result is **up to scale** because the recovered translation is unit-length.
  Set `DepthReconstructionConfig::metricBaseline` to the real baseline (in meters,
  computable from the ETH3D `images.txt` poses) for metric depth.
* Auto-orients the cloud so depth is positive (handles left/right-swapped inputs).
* Clips far outliers at `maxDepthPercentile` (default 98th percentile) so the mesh
  stays viewable without a hand-tuned absolute `maxDepth`. The reconstruction is
  only up to scale, so an absolute limit is arbitrary and drifts between runs; the
  percentile adapts automatically. Set an explicit `maxDepth` to override.
* Exports a colored point cloud and a triangle mesh (neighboring valid pixels are
  triangulated; quads spanning a depth discontinuity are skipped to avoid stretched
  faces). Poisson meshing can be added later.

## Custom Geometry Implementation

As part of replacing OpenCV building blocks, the project now includes a custom
fundamental-matrix estimation module.

### Custom normalized 8-point algorithm

Implemented in:

```text
include/CustomEightPoint.h
src/CustomEightPoint.cpp

## Current Notes

* StereoBM and StereoSGBM are both available for disparity-map generation.
* StereoSGBM generally provides denser disparity maps. All of its knobs
  (`blockSize`, `p1`/`p2`, `uniquenessRatio`, `speckleWindowSize`/`speckleRange`,
  `disp12MaxDiff`, `preFilterCap`) are exposed in `DenseStereoConfig` for tuning,
  plus a `medianKernel` post-filter. On full 25 MP ETH3D images SGBM is slow and
  memory-heavy; consider downscaling while iterating.
* **WLS filtering** (`useWlsFilter`, ximgproc) runs a left+right matcher and
  edge-aware smoothing guided by the rectified left image — the single biggest
  noise reducer. It also fills textureless/occluded regions by extrapolation, so
  `wlsConfidenceThreshold` (default 0.5) drops the low-confidence (guessed) pixels
  so coverage reflects actually-matched geometry. Lower it (or set 0) for a denser
  but partly-interpolated map; raise it for a sparser, more trustworthy one.
* `numDisparities` (default 320) must cover the scene's disparity range; the
  matcher prints the observed range and warns if it hits the search bound. A few
  border pixels (the left strip with no right-image overlap) always saturate — the
  confidence gate removes them.
* The recovered translation vector currently provides direction only, so reconstruction is up to scale. Metric depth requires the known camera baseline (from `images.txt`).
* **Left/right ordering:** disparity matching assumes the second camera is to the right (`t.x < 0`). `main` now checks the recovered translation and, if `t.x > 0`, swaps the pair and recomputes the geometry so disparities are correct (on `delivery_area` this raised valid-disparity coverage from ~13% to ~20%). `DepthReconstructor` also keeps a cloud-orientation safety net.
* **Known issue to address:** `DenseStereoMatcher` calls `StereoBM::compute(right, left)` but `StereoSGBM::compute(left, right)` — the inconsistent argument order flips the disparity sign for the BM path (the SGBM path used by default is correct).
* Per the TA, the OpenCV building blocks (8-point algorithm, rectification, dense matching, depth estimation) are to be replaced with our own custom implementations.
