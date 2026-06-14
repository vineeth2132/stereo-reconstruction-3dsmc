# Stereo Reconstruction

A C++ stereo reconstruction pipeline implemented with OpenCV and Eigen.

The current pipeline performs:

1. Stereo image loading
2. Camera intrinsics loading
3. SIFT keypoint detection
4. Sparse feature matching with ratio filtering
5. Fundamental matrix estimation with RANSAC
6. Essential matrix computation
7. Relative camera pose recovery
8. Stereo rectification
9. Dense disparity-map generation with StereoBM or StereoSGBM
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
| `disparity_sgbm.png` | Colored disparity map (invalid pixels black) |
| `pointcloud_sgbm.ply` | Colored 3D point cloud |
| `mesh_sgbm.ply` | Triangle mesh from the disparity grid |

Open the `.ply` files in MeshLab or CloudCompare. `out/` is git-ignored.

## Depth reconstruction (`DepthReconstructor`)

`DepthReconstructor` turns the disparity map into 3D geometry:

* Uses the rectification `Q` matrix (`cv::reprojectImageTo3D`) to back-project each
  valid disparity pixel into a 3D point, colored from the rectified left image.
* The result is **up to scale** because the recovered translation is unit-length.
  Set `DepthReconstructionConfig::metricBaseline` to the real baseline (in meters,
  computable from the ETH3D `images.txt` poses) for metric depth.
* Auto-orients the cloud so depth is positive (handles left/right-swapped inputs).
* Exports a colored point cloud and a triangle mesh (neighboring valid pixels are
  triangulated; quads spanning a depth discontinuity are skipped to avoid stretched
  faces). Poisson meshing can be added later.

## Current Notes

* StereoBM and StereoSGBM are both available for disparity-map generation.
* StereoSGBM generally provides denser disparity maps, but its parameters still need tuning. On full 25 MP ETH3D images SGBM is slow and memory-heavy; consider downscaling while iterating.
* The recovered translation vector currently provides direction only, so reconstruction is up to scale. Metric depth requires the known camera baseline (from `images.txt`).
* **Left/right ordering:** disparity matching assumes the second camera is to the right (`t.x < 0`). `main` now checks the recovered translation and, if `t.x > 0`, swaps the pair and recomputes the geometry so disparities are correct (on `delivery_area` this raised valid-disparity coverage from ~13% to ~20%). `DepthReconstructor` also keeps a cloud-orientation safety net.
* **Known issue to address:** `DenseStereoMatcher` calls `StereoBM::compute(right, left)` but `StereoSGBM::compute(left, right)` — the inconsistent argument order flips the disparity sign for the BM path (the SGBM path used by default is correct).
* Per the TA, the OpenCV building blocks (8-point algorithm, rectification, dense matching, depth estimation) are to be replaced with our own custom implementations.
