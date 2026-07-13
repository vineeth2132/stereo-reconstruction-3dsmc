# Stereo Reconstruction

A C++ stereo reconstruction pipeline implemented with OpenCV and Eigen.

The current pipeline performs:

1. Stereo image loading
2. Camera intrinsics loading
3. SIFT keypoint detection
4. Sparse feature matching with ratio filtering
5. Fundamental matrix estimation using a custom Hartley-normalized 8-point algorithm with custom Sampson-error RANSAC (`CustomEightPoint`), with `cv::findFundamentalMat` kept only for comparison/debugging
6. Essential matrix computation
7. Relative camera pose recovery
8. Stereo rectification
9. Dense disparity-map generation, via one of two interchangeable backends (see
   [Dense matching backends](#dense-matching-backends)):
   * **OpenCV** — StereoBM/StereoSGBM, optionally refined with an edge-aware WLS
     filter (`opencv_contrib` ximgproc) and a median + WLS confidence post-filter.
   * **Custom** — our own hand-written NCC block matcher (no OpenCV stereo class).
10. Depth reconstruction: disparity → 3D points via the rectification `Q` matrix
11. Colored point-cloud export (`.ply`)
12. Triangle-mesh export (`.ply`) from the disparity grid, with depth-discontinuity culling

Each stage also writes a step output to the `out/` directory (see [Outputs](#outputs))
so the pipeline can be inspected without the blocking `imshow` debug windows. `main`
runs the dense → depth → mesh tail once per backend and tags the outputs (`opencv`,
`custom`) so the two can be compared directly.

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

The `scripts/visualize_maps.py` figure helper uses the project venv with a few
extra packages:

```powershell
.venv\Scripts\python.exe -m pip install numpy matplotlib tifffile imagecodecs
```

(`imagecodecs` is needed because OpenCV writes the mask `.tiff` files LZW-compressed.)

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

Running the executable writes the following step outputs to `out/`. The dense and
3D outputs are produced once per backend and carry a `<tag>` of `opencv` or
`custom`:

| File | Description |
|------|-------------|
| `rectified_left.png`, `rectified_right.png` | Rectified pair — corresponding points should lie on the same rows |
| `disparity_<tag>_colored.png` | Colored disparity map (invalid pixels black) |
| `disparity_<tag>_float.tiff`, `valid_disparity_<tag>_mask.tiff` | Raw disparity + mask (for quantitative inspection) |
| `depth_<tag>_float.tiff`, `valid_depth_<tag>_mask.tiff` | Per-pixel depth (Z) + mask |
| `depth_dust3r_float.tiff`, `valid_depth_dust3r_mask.tiff` | Metric DUSt3R depth map + valid mask |
| `pointcloud_<tag>.ply` | Colored 3D point cloud |
| `mesh_<tag>.ply` | Triangle mesh from the disparity grid |
| `valid_matched_custom_mask.tiff` | Custom backend only: the pre-fill matched mask (honest, fill-free coverage; the ~36% of valid pixels that were genuinely matched vs. directionally filled) |
| `reason_custom.png` | Custom backend only: raw indexed 0..4 pre-fill failure-reason map (see `CustomMatchReason`; colored/tabulated by `evaluate_depth.py`) |
| `pointcloud_custom_matched.ply`, `mesh_custom_matched.ply` | Custom matched-only 3D outputs — the reconstruction restricted to `validMask ∧ matchedMask` (fill-free, crisp; smaller than the dense `*_custom.ply`) |
| `disparity_<tag>_scaled.png`, `depth_<tag>_scaled.png` | Color-scaled map with colorbar, **white = invalid** (from `visualize_maps.py`) |
| `disparity_custom_matched_scaled.png` | Custom disparity masked to matched-only pixels, same color scale (from `visualize_maps.py`) |
| `comparison_disparity.png`, `comparison_depth.png` | Four-panel side-by-side (opencv \| custom dense, filled \| custom matched-only \| ground truth), shared color scale; panels degrade gracefully if the GT or matched mask is missing (from `visualize_maps.py`) |
| `gt_depth_raw_float.tiff`, `gt_depth_rectified_float.tiff` | ETH3D ground-truth depth (raw + rectified into the disparity frame), written only when the GT data is present |
| `error_<tag>_scaled.png` | Absolute depth-error map vs. rectified GT, shared color scale, white = invalid (from `evaluate_depth.py`) |
| `error_dust3r_scaled.png` | DUSt3R absolute depth-error map vs. aligned rectified GT |

The dense/3D outputs are only produced in their `<tag>`ged form; the earlier
un-tagged `mesh.ply`, `pointcloud.ply`, and `disparity_*` names are no longer
written.

Generate the scaled / comparison figures with the Python helper (after running the
executable):

```powershell
.venv\Scripts\python.exe scripts\visualize_maps.py
```

It mirrors the MATLAB `scripts/visualize_disparity.m` / `visualize_depth.m`
helpers (color scale, invalid shown white) but renders both backends and the
side-by-side comparison in one go.

Open the `.ply` files in MeshLab or CloudCompare. `out/` is git-ignored.

## DUSt3R depth estimation

An optional DUSt3R-based depth backend is available through
`scripts/run_dust3r_depth.py`. It runs the official two-view `PairViewer`
workflow on `out/rectified_left.png` and `out/rectified_right.png`.

DUSt3R predicts depth up to an unknown global scale. The script converts the
prediction to metric depth by matching the predicted camera baseline to the
physical ETH3D stereo baseline.

DUSt3R must be cloned separately next to this repository:

```text
3d_project/
├── stereo-reconstruction-3dsmc/
└── dust3r/
```

Run it from the project root with the DUSt3R virtual environment active:

```bash
source .venv-dust3r/bin/activate

python scripts/run_dust3r_depth.py \
  --dust3r-root ../dust3r \
  --baseline 0.4289
```

The script writes:

```text
out/depth_dust3r_float.tiff
out/valid_depth_dust3r_mask.tiff
```

Evaluate it together with the other backends:

```bash
python scripts/evaluate_depth.py --tags opencv custom dust3r
```

DUSt3R uses a resized and center-cropped input resolution, so the evaluator
applies the same alignment to the ground-truth depth before computing its
metrics.

## Quantitative evaluation

`scripts/evaluate_depth.py` scores each backend's metric depth map against the
rectified ETH3D ground truth. For every tag it intersects our-valid with GT-valid
and reports coverage, MAE, RMSE, median absolute error, bad-pixel ratios (at
0.1 / 0.5 / 1.0 m), and a median-scale-aligned MAE that removes any global scale
bias so structural error can be inspected on its own. It prints a comparison table
and writes an absolute-error map per backend (`error_<tag>_scaled.png`).

It needs the ground-truth depth downloaded (`--with-depth`) so the pipeline emits
`gt_depth_rectified_float.tiff` (the GT block in `main` is wrapped in try/catch, so
the pipeline still runs without it), and the same venv packages as
`visualize_maps.py`:

```powershell
.venv\Scripts\python.exe scripts\download_eth3d.py --with-depth
.\build\Release\stereo_reconstruction.exe
.venv\Scripts\python.exe scripts\evaluate_depth.py
```

On `delivery_area` (metric baseline 0.4289 m recovered from `images.txt`), coverage is
reported as a fraction of the GT-valid region. The custom backend is reported in two
variants: **dense** (after occlusion-aware hole filling — high coverage, but ~60% of its
valid pixels are directional-fill interpolations) and **matched-only** (the fill-free
subset the matcher genuinely matched — lower coverage but much more accurate):

| Metric | OpenCV (tuned) | Custom (dense) | Custom (matched-only) |
|--------|---------------:|---------------:|----------------------:|
| coverage | 31.7% | **91.9%** | 36.3% |
| MAE | 0.0858 m | 0.2243 m | **0.0397 m** |
| median AE | — | 0.0304 m | **0.0179 m** |
| bad > 0.5 m | — | 7.6% | — |
| bad > 2 px | — | 27.8% | **7.1%** |

The dense fill trades accuracy for coverage: it fills the textureless floor and occluded
strips so the map is continuous, but those filled pixels drag the MAE up. The matched-only
row is the honest measurement of the matcher itself and beats tuned OpenCV SGBM on every
shared metric. Coverage tops out around ~92% because the left/right image bands are never
seen by the second camera, so no amount of filling can recover them.

## Dense matching backends

The dense stage is split behind the `IDenseStereoMatcher` interface, so the OpenCV
matcher and our own custom matcher are drop-in interchangeable — everything
downstream only depends on the resulting `DenseMatchingResult`. Select with one
flag, `DenseStereoConfig::backend`:

* `DenseStereoBackend::OpenCv` (`DenseStereoMatcher`) — OpenCV StereoBM/StereoSGBM
  with the optional WLS refinement described above.
* `DenseStereoBackend::Custom` (`CustomDenseMatcher`) — **our own** hierarchical
  coarse-to-fine normalized cross-correlation (NCC) block matcher. It builds an
  image pyramid (from `customCoarsestDownscale` ≈ 0.0625 up to
  `customFinalDownscale` = 0.5) and does a **full, width-derived disparity search**
  on the tiny coarsest level (no hand-tuned `numDisparities`); each finer level then
  warps the target image by the upsampled prior disparity (`cv::remap`) and searches
  only a small `±customResidualRadius` residual around it. It builds the cost volume,
  winner-take-all disparity, parabola subpixel refinement, a full-pyramid left-right
  consistency check, and a correlation-confidence gate itself; it does **not** use
  any OpenCV stereo class. OpenCV is used only for box-filter window sums,
  resize/remap and image arithmetic as numerical primitives. NCC is invariant to
  brightness/contrast differences between the two cameras, which the raw SAD/SSD cost
  is not.

`CreateDenseMatcher(backend)` returns the right implementation. `main` currently
runs **both** so the maps can be compared; to use just one, instantiate the single
backend you want.

Deriving the disparity range from image geometry means the custom matcher needs no
`numDisparities` bound: the coarsest level searches `round(coarsest_width *
customMaxDisparityFraction)` (default 1/3 of the width) and the pyramid refines from
there, so the final disparity is produced at `customFinalDownscale` and then
upscaled/rescaled to full resolution, keeping `DepthReconstructor` unchanged. After
matching, a valid-aware median (`customMedianKernel`, over valid neighbours only) and
our own connected-component speckle filter (`customSpeckleMinArea`,
`customSpeckleTolerance`) clean the map at working resolution. Its knobs
(`customFinalDownscale`, `customCoarsestDownscale`, `customWindowSize`,
`customResidualRadius`, `customMaxDisparityFraction`, `customLrConsistency`,
`customMinCorrelation`, `customSubpixel`, `customMedianKernel`,
`customSpeckleMinArea`, `customSpeckleTolerance`) live in `DenseStereoConfig`.

### Occlusion-aware densification

The speckle-filtered map is accurate but sparse (~36% of the GT region). A three-stage
densification pass then fills the holes without disturbing the genuinely matched pixels:

1. **`FillHolesDirectional`** (`customFillHoles`) — an SGM-style directional fill. For
   each hole pixel it scans the 8 compass directions for the nearest valid disparity;
   when the failure reason is an LR-occlusion it takes the **second-smallest** of the
   directional candidates (Hirschmüller's occlusion rule — occlusions belong to the
   background, so the smallest/nearest value is skipped), otherwise the **median**. A
   pixel is only filled when a valid disparity is reachable in at least
   `customFillMinDirections = 5` of the 8 directions: the left no-correspondence band
   (~20% of the GT region) has exactly 5 directions available, so 5 fills it while 6
   would leave it (and other genuinely-unobserved strips) permanently blank.
2. **`GuidedDiffusionRefine`** (`customGuidedFillIterations`) — **ships disabled (0)**.
   It relaxed mismatch-type fills toward the surrounding measurements via a guide-weighted
   normalized convolution, but in textureless holes the guide has no image edges, so it
   degenerated to plain averaging and smeared depth across discontinuities that carry no
   edge (MAE 0.220 → 0.362 m on `delivery_area`). Kept as an ablation, off by default.
3. **`WeightedMedianRefine`** (`customWeightedMedianRadius = 9`,
   `customWeightedMedianIterations = 2`) — an edge-aware weighted median applied to the
   **filled pixels only**, guided by the rectified left image so the fill follows image
   edges instead of crossing them. Two passes at radius 9 propagate edges into the filled
   regions; matched pixels are left untouched.

The densification produces the **dense** map. To keep the interpolated fills honest, the
matcher also records, before filling, `matchedMask` (the genuinely-matched pixels) and
`failureReason` (per-pixel `CustomMatchReason` codes: matched / never-matched / border /
LR-rejected / speckle). `main` writes these as `valid_matched_custom_mask.tiff` and
`reason_custom.png`, and — for the custom backend — additionally exports the fill-free
**matched-only** 3D outputs (`pointcloud_custom_matched.ply`, `mesh_custom_matched.ply`)
by restricting the reconstruction to `validMask ∧ matchedMask`. `visualize_maps.py` renders
a matched-only comparison panel next to the dense and GT panels so the two can be judged
side by side, and `evaluate_depth.py` reports a separate `custom-matched` row.

## Depth reconstruction (`DepthReconstructor`)

`DepthReconstructor` turns the disparity map into 3D geometry:

* Back-projects each valid disparity pixel into a 3D point via one of two backends,
  selected by `DepthReconstructionConfig::backend`:
  * `DepthBackend::Custom` (default) — **our own** per-pixel homogeneous
    back-projection `[X,Y,Z,W] = Q · [x, y, d, 1]` (equivalently `Z = f·B/d`). When
    `validateAgainstOpenCv` is set it also runs `cv::reprojectImageTo3D` and logs the
    max absolute component difference as a sanity check (the two agree to ~1e-4).
  * `DepthBackend::OpenCv` — `cv::reprojectImageTo3D`, kept only for comparison.
  Points are colored from the rectified left image.
* The result is **up to scale** because the recovered translation is unit-length.
  Set `DepthReconstructionConfig::metricBaseline` to the real baseline (in meters,
  computable from the ETH3D `images.txt` poses) for metric depth.
* Auto-orients the cloud so depth is positive (handles left/right-swapped inputs).
* Clips far outliers at `maxDepthPercentile` (default 99.9th percentile) so the mesh
  stays viewable without a hand-tuned absolute `maxDepth`. The reconstruction is
  only up to scale, so an absolute limit is arbitrary and drifts between runs; the
  percentile adapts automatically. The default was 98 but that clipped real far
  geometry (with the dense filled disparity the far tail is genuine background, and
  depth is already bounded by `fB / min(disparity)` ≈ 28 m on `delivery_area`); 98
  cut GT-verified pixels between 15.3 m and 18.3 m, so it was raised to 99.9. Set an
  explicit `maxDepth` to override.
* Exports a colored point cloud and a triangle mesh (neighboring valid pixels are
  triangulated; quads spanning a depth discontinuity are skipped to avoid stretched
  faces). The `.ply` files are written as **binary little-endian**, and
  `exportGridStep` (default 2 in `main`) strides the point-cloud/mesh grid — since
  the custom matcher runs at 0.5 scale, step 2 is effectively lossless and keeps the
  dense `mesh_custom.ply` around ~220 MB (the fill-free `mesh_custom_matched.ply` is
  ~67 MB) instead of the multi-GB an ASCII full-grid export would produce. Poisson
  meshing can be added later.

## Custom Geometry Implementation

As part of replacing OpenCV building blocks, the project now includes a custom
fundamental-matrix estimation module.

### Custom normalized 8-point algorithm

A Hartley-normalized 8-point solver wrapped in a Sampson-error RANSAC loop is the
primary fundamental-matrix path; `cv::findFundamentalMat` is retained only for
comparison logging. Implemented in:

```text
include/CustomEightPoint.h
src/CustomEightPoint.cpp
```

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
* `numDisparities` (default 704) must cover the scene's disparity range; the
  matcher prints the observed range and warns if it hits the search bound.
  `delivery_area`'s true disparity range reaches ~690 full-res px, so the previous
  320 clamped all the near geometry. A few border pixels (the left strip with no
  right-image overlap) always saturate — the confidence gate removes them.
* **Custom NCC matcher** (`DenseStereoBackend::Custom`): the hierarchical
  coarse-to-fine search recovers the same scene structure / near-far gradient as the
  OpenCV map and, because the coarsest level derives its disparity range from the
  image width, needs no hand-tuned search bound to reach the nearest geometry. It can
  still be noisier on textureless regions such as the floor, where NCC is unreliable.
  Tighten `customMinCorrelation` and/or `customLrConsistency`, raise
  `customWindowSize`, or grow the valid-aware post-filters (`customMedianKernel`,
  `customSpeckleMinArea`) to trade density for cleanliness; lower
  `customFinalDownscale` (0.5 → 0.25) if the 25 MP pairs are too slow.
* The recovered translation vector currently provides direction only, so reconstruction is up to scale. Metric depth requires the known camera baseline (from `images.txt`).
* **Left/right ordering:** disparity matching assumes the second camera is to the right (`t.x < 0`). `main` now checks the recovered translation and, if `t.x > 0`, swaps the pair and recomputes the geometry so disparities are correct (on `delivery_area` this raised valid-disparity coverage from ~13% to ~20%). `DepthReconstructor` also keeps a cloud-orientation safety net.
* **Known issue to address:** `DenseStereoMatcher` calls `StereoBM::compute(right, left)` but `StereoSGBM::compute(left, right)` — the inconsistent argument order flips the disparity sign for the BM path (the SGBM path used by default is correct).
* Per the TA, the OpenCV building blocks (8-point algorithm, rectification, dense matching, depth estimation) are to be replaced with our own custom implementations.
