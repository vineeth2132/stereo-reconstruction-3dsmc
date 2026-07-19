# Stereo Reconstruction

A C++ stereo reconstruction pipeline implemented with OpenCV and Eigen, compared
against two pretrained deep methods (RAFT-Stereo, DUSt3R) on ETH3D scenes.

**All four TA-required components are custom implementations** — OpenCV is used
only for I/O, SIFT, and numerical primitives (box filters, resize/remap, image
arithmetic), plus as the tuned SGBM baseline the custom code is compared against:

* 8-point fundamental-matrix estimation + Sampson RANSAC (`CustomEightPoint`)
* stereo rectification (`StereoRectifier::CustomStereoRectify` + own
  rectification maps and bilinear remap)
* dense matching (`CustomDenseMatcher`, no OpenCV stereo class)
* depth estimation (`DepthReconstructor`, own per-pixel `Q` back-projection)

The pipeline performs:

1. Stereo image loading
2. Camera intrinsics loading
3. SIFT keypoint detection
4. Sparse feature matching with ratio filtering
5. Fundamental matrix estimation using a custom Hartley-normalized 8-point algorithm with custom Sampson-error RANSAC (`CustomEightPoint`), with `cv::findFundamentalMat` kept only for comparison/debugging
6. Essential matrix computation
7. Relative camera pose recovery
8. Custom stereo rectification (rectifying rotations from the recovered pose, own rectification maps + bilinear remap)
9. Dense disparity-map generation, via interchangeable backends (see
   [Dense matching backends](#dense-matching-backends)):
   * **OpenCV** — StereoBM/StereoSGBM, optionally refined with an edge-aware WLS
     filter (`opencv_contrib` ximgproc) and a median + WLS confidence post-filter.
   * **Custom** — our own hierarchical coarse-to-fine matcher with three
     swappable matching costs: **NCC**, **SSD**, and **Census** (no OpenCV
     stereo class).
10. Depth reconstruction: disparity → 3D points via the rectification `Q` matrix
11. Colored point-cloud export (`.ply`)
12. Triangle-mesh export (`.ply`) from the disparity grid, with depth-discontinuity culling

Each stage also writes a step output to the output directory (see
[Outputs](#outputs)) so the pipeline can be inspected without the blocking
`imshow` debug windows. `main` runs the dense → depth → mesh tail once per
configuration and tags the outputs: `opencv` (SGBM+WLS), and `NCC` / `SSD` /
`Census` for the custom matcher. Every custom run additionally exports its
three post-processing stages — `raw` (after left-right consistency), `filtered`
(after valid-aware median + speckle removal), and `filled` (after
occlusion-aware densification) — so accuracy and coverage can be judged
separately at every step.

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

### Python environments

Two virtual environments cover all the Python tooling (Windows commands shown;
use `python3 -m venv` on Linux):

**`.venv`** — dataset download, visualization, and evaluation
(`download_eth3d.py`, `visualize_maps.py`, `evaluate_depth.py`):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install numpy matplotlib tifffile imagecodecs opencv-python-headless py7zr
```

(`imagecodecs` is needed because OpenCV writes the mask `.tiff` files LZW-compressed.)

**`.venv-raft`** — one shared environment for both deep methods (RAFT-Stereo
and DUSt3R). A CUDA-capable GPU is recommended but modest: everything below runs
on a 6 GB GTX 1660 Ti (RAFT inference at 640×416 takes ~1 s):

```powershell
py -3.12 -m venv .venv-raft
.venv-raft\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
.venv-raft\Scripts\python.exe -m pip install numpy opencv-python matplotlib tqdm scipy tifffile imageio scikit-image opt_einsum roma einops trimesh "huggingface-hub[torch]>=0.22" safetensors
```

See [RAFT-Stereo depth estimation](#raft-stereo-depth-estimation) and
[DUSt3R depth estimation](#dust3r-depth-estimation) for the model checkouts.

## Project Structure

```text
<parent directory>/
├── stereo-reconstruction/
│   ├── CMakeLists.txt
│   ├── vcpkg.json
│   ├── include/
│   ├── src/
│   ├── scripts/        # download / visualization / evaluation / deep-method wrappers
│   ├── data/           # input images + calibration (images are git-ignored)
│   ├── out/            # step outputs, default scene (git-ignored)
│   ├── out_<scene>/    # step outputs for additional scenes (git-ignored)
│   ├── third_party/    # RAFT-Stereo clone + checkpoints (git-ignored)
│   ├── .venv/          # Python tooling venv (git-ignored)
│   ├── .venv-raft/     # deep-methods venv (git-ignored)
│   └── build/          # generated build files (git-ignored)
└── dust3r/             # sibling DUSt3R clone (see DUSt3R section)
```

The `build/` directory contains generated build files and should not be committed.

## Dataset

The pipeline runs on **ETH3D high-resolution multi-view (DSLR, undistorted) scenes**.
ETH3D is non-rectified, which is required for this project. Evaluated scenes so
far: `delivery_area` (default) and `facade`.

The image files are **not committed** (they are git-ignored); only `data/delivery_area/dslr_calibration_undistorted/cameras.txt` is tracked. Download a scene with the helper script (uses the project venv, no 7-Zip needed):

```powershell
.venv\Scripts\python.exe scripts\download_eth3d.py              # delivery_area images + calibration (~0.5 GB)
.venv\Scripts\python.exe scripts\download_eth3d.py --with-depth # also ground-truth depth (for evaluation)

# another scene, optionally to a different drive:
.venv\Scripts\python.exe scripts\download_eth3d.py --scene facade --with-depth --dest D:/eth3d-data
```

This extracts to `<dest>/<scene>/`; the scene directory is passed to the
executable as its first argument (default `../data/delivery_area/`).

One extra step for scenes other than `delivery_area`: the ground-truth
rectification also needs the scene's **raw-frame calibration**
(`<scene>/dslr_calibration_jpg/cameras.txt`), which ETH3D ships inside the
separate `<scene>_dslr_jpg.7z` archive — download it and extract just the
`dslr_calibration_jpg/` folder next to the others (the delivery_area copy is
already in the repo's `data/` layout).

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

Run (from inside `build/` — the default paths are relative to it):

```powershell
.\Release\stereo_reconstruction.exe                # default: delivery_area DSC_0688/0689 -> ../out
```

The executable is scene-parameterizable:

```text
stereo_reconstruction [sceneDir] [leftImage] [rightImage] [outDir]
```

* `sceneDir` — ETH3D scene root (default `../data/delivery_area/`)
* `leftImage` / `rightImage` — image paths relative to `sceneDir`
* `outDir` — output directory (default `../out`)
* accepted argument counts: 0, 1 (sceneDir), 3 (sceneDir + pair), 4 (all); `--help` prints usage

Example — the `facade` scene into its own output directory:

```powershell
.\Release\stereo_reconstruction.exe D:/eth3d-data/facade images/dslr_images_undistorted/DSC_0336.JPG images/dslr_images_undistorted/DSC_0337.JPG ../out_facade
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

Running the executable writes the following step outputs to the output directory
(`out/` by default). The custom dense/3D outputs are produced once per cost
metric `<tag>` ∈ {`NCC`, `SSD`, `Census`} **and** per post-processing stage
`<stage>` ∈ {`raw`, `filtered`, `filled`}; the OpenCV baseline uses tag
`opencv` with the single stage `final`:

| File | Description |
|------|-------------|
| `rectified_left.png`, `rectified_right.png` | Rectified pair — corresponding points should lie on the same rows |
| `rectified_meta.txt` | Rectified-frame metadata: `fx`, `baseline`, `width`, `height` — auto-read by the RAFT/DUSt3R wrappers (the rectified fx differs from the raw `cameras.txt` fx!) |
| `disparity_<tag>_<stage>_float.tiff`, `valid_disparity_<tag>_<stage>_mask.tiff` | Custom disparity + validity mask per stage |
| `disparity_<tag>_<stage>_colored.png` | Colored disparity map (invalid pixels black) |
| `depth_<tag>_<stage>_float.tiff`, `valid_depth_<tag>_<stage>_mask.tiff` | Metric per-pixel depth (Z) + mask |
| `pointcloud_<tag>_<stage>.ply`, `mesh_<tag>_<stage>.ply` | Colored 3D point cloud and triangle mesh per stage |
| `disparity_opencv_float.tiff`, `valid_disparity_opencv_mask.tiff` | OpenCV disparity + mask (single stage) |
| `depth_opencv_final_float.tiff`, `pointcloud_opencv_final.ply`, `mesh_opencv_final.ply`, … | OpenCV depth/3D outputs, stage tag `final` |
| `reason_<tag>.png` | Custom runs: indexed 0..4 pre-fill failure-reason map (see `CustomMatchReason`) |
| `gt_depth_raw_float.tiff`, `gt_depth_rectified_float.tiff` | ETH3D ground-truth depth (raw + rectified into the disparity frame), written only when the GT data is present |
| `disparity_raft_float.tiff`, `depth_raft_float.tiff`, `valid_depth_raft_mask.tiff`, `disparity_raft.npy` | RAFT-Stereo outputs at its native inference resolution (from `run_raft_depth.py`) |
| `depth_dust3r_float.tiff`, `valid_depth_dust3r_mask.tiff` | Metric DUSt3R depth map + valid mask (from `run_dust3r_depth.py`) |
| `error_<tag>_scaled.png` | Scale-aligned absolute depth-error map vs. rectified GT, shared color scale, white = invalid (from `evaluate_depth.py`) |

The `raw` → `filtered` → `filled` triplet makes the accuracy/coverage tradeoff
explicit: `filtered` is the fill-free measurement of the matcher itself, while
`filled` is the dense final product whose holes were interpolated.

Generate the scaled / comparison figures with the Python helper (after running the
executable):

```powershell
.venv\Scripts\python.exe scripts\visualize_maps.py
```

It mirrors the MATLAB `scripts/visualize_disparity.m` / `visualize_depth.m`
helpers (color scale, invalid shown white) but renders every method and stage
(`--methods NCC SSD Census`, `--stages raw filtered filled`) plus per-stage
side-by-side comparison panels (`comparison_<kind>_<stage>.png`) in one go.

Open the `.ply` files in MeshLab or CloudCompare. `out/` is git-ignored.

## RAFT-Stereo depth estimation

`scripts/run_raft_depth.py` runs pretrained RAFT-Stereo on the rectified pair
written by the C++ pipeline (`out/rectified_left.png` / `rectified_right.png`),
then converts the predicted disparity to metric depth via `depth = fx·B/d`
using the rectified-frame focal length and baseline from
`out/rectified_meta.txt` (explicit `--fx` / `--baseline` flags override).

One-time setup — clone the official repo into `third_party/` (git-ignored) and
fetch the checkpoints (the Dropbox URL is in the repo's `download_models.sh`;
the wrapper expects `models/raftstereo-eth3d.pth`):

```powershell
git clone https://github.com/princeton-vl/RAFT-Stereo third_party/RAFT-Stereo
# download + unzip models.zip from download_models.sh into third_party/RAFT-Stereo/models/
```

Run (after the C++ pipeline, so the rectified pair + meta file exist):

```powershell
.venv-raft\Scripts\python.exe scripts\run_raft_depth.py --width 640
# other scenes: add e.g. --out ..\out_facade
```

The pair is downscaled to `--width` (default 640, both dimensions floored to
multiples of 32) for inference; the outputs stay at that native resolution and
the evaluator upsamples them to the ground-truth frame (see below). It writes
`disparity_raft_float.tiff`, `depth_raft_float.tiff`,
`valid_depth_raft_mask.tiff`, and `disparity_raft.npy` into the output
directory.

**Note on fx:** the rectified frame's focal length (~4330 px on
`delivery_area`) is *not* the raw `cameras.txt` fx (3408.59 px) — the custom
rectification synthesizes a new projection. Using the raw value biased RAFT's
metric depth by 1.27×, which is why the wrapper reads `rectified_meta.txt`.

## DUSt3R depth estimation

`scripts/run_dust3r_depth.py` runs the official two-view `PairViewer` workflow
on the same rectified pair. DUSt3R predicts geometry up to an unknown global
scale; the script converts to metric depth by matching the predicted camera
baseline to the physical baseline from `rectified_meta.txt` (or `--baseline`).

DUSt3R must be cloned separately **next to** this repository:

```powershell
git clone --recursive https://github.com/naver/dust3r ..\dust3r
```

Run with the shared deep-methods venv (the ViT-Large checkpoint, ~2.6 GB,
auto-downloads from Hugging Face on first use — set `$env:HF_HOME` to another
drive first if `C:` is tight):

```powershell
.venv-raft\Scripts\python.exe scripts\run_dust3r_depth.py
# other scenes: add e.g. --out ..\out_facade
```

It writes `depth_dust3r_float.tiff` and `valid_depth_dust3r_mask.tiff` at
DUSt3R's native resolution (512×336 for our 3:2 pairs: long edge scaled to 512,
center-cropped vertically). The evaluator inverts exactly that resize+crop when
mapping the prediction back onto the full-resolution ground truth.

## Quantitative evaluation

`scripts/evaluate_depth.py` scores every method against the rectified ETH3D
ground truth using the **TA-prescribed resolution-fairness convention**: each
method runs at its own native resolution (classical C++: 6208×4135;
RAFT-Stereo: 640×416; DUSt3R: 512×336), its *output* is upsampled
(nearest-neighbor) to the full-resolution ground-truth frame, and **all metrics
are computed there**. The script prints each tag's native resolution for
transparency. For DUSt3R the long-edge-512 resize + vertical center crop is
inverted exactly (rows DUSt3R never saw stay invalid); RAFT disparities are
scaled into full-resolution pixel units when upsampled.

Per tag it reports valid-pixel count, coverage of the GT-valid region, MAE,
RMSE, median absolute error, bad-pixel ratios (0.1 / 0.5 / 1.0 m), disparity
bad-pixel ratios (1 / 2 / 4 px, via `fB = median(disp·depth)`), and
median-scale-aligned variants of the depth metrics (removes any global scale
bias so structural error can be inspected on its own). It writes one
scale-aligned absolute-error map per tag (`error_<tag>_scaled.png`).

It needs the ground-truth depth downloaded (`--with-depth`) so the pipeline emits
`gt_depth_rectified_float.tiff` (the GT block in `main` is wrapped in try/catch, so
the pipeline still runs without it):

```powershell
.venv\Scripts\python.exe scripts\download_eth3d.py --with-depth
.\build\Release\stereo_reconstruction.exe
.venv\Scripts\python.exe scripts\evaluate_depth.py                      # default: all tags, out/
.venv\Scripts\python.exe scripts\evaluate_depth.py --out ..\out_facade  # other scenes
```

The default `--tags` covers `opencv_final`, `NCC/SSD/Census × raw/filtered/filled`,
`dust3r`, and `raft`; missing files are skipped with a message.

### Results — `delivery_area` (DSC_0688/0689, baseline 0.4289 m, GT-valid 988,428 px)

| Method (stage) | native res | coverage | MAE (m) | median AE (m) | bad > 2 px |
|---|---|---:|---:|---:|---:|
| OpenCV SGBM+WLS (final) | 6208×4135 | 33.2% | 0.0850 | 0.0408 | 32.2% |
| NCC raw | 6208×4135 | 38.1% | 0.0537 | 0.0209 | 8.7% |
| NCC filtered | 6208×4135 | 36.4% | 0.0416 | 0.0192 | 6.5% |
| NCC filled | 6208×4135 | 93.2% | 0.2315 | 0.0326 | 29.2% |
| SSD raw | 6208×4135 | 11.2% | 0.1366 | 0.0359 | 23.6% |
| SSD filtered | 6208×4135 | 9.9% | 0.0775 | 0.0314 | 17.8% |
| SSD filled | 6208×4135 | 64.2% | 0.5806 | 0.1150 | 55.9% |
| Census raw | 6208×4135 | 52.3% | 0.0389 | 0.0156 | 6.1% |
| Census filtered | 6208×4135 | **51.2%** | **0.0331** | **0.0149** | **4.9%** |
| Census filled | 6208×4135 | 93.3% | 0.2155 | 0.0257 | 20.9% |
| DUSt3R | 512×336 | 99.2% | 0.3297 | 0.2340 | — |
| RAFT-Stereo | 640×416 | **100.0%** | **0.1222** | 0.0355 | 38.7% |

Takeaways:

* **Census is the strongest classical cost on every axis** — its rank-based
  descriptor is robust to the illumination differences between the two DSLR
  shots, giving it both the best accuracy and the highest fill-free coverage.
* **SSD is the weakest**, and its `filled` row is instructive: densification
  quality depends directly on anchor density — interpolating from 11% coverage
  produces a 0.58 m MAE, so filling amplifies whatever the matcher provides.
* The `raw → filtered → filled` progression makes the accuracy ↔ coverage
  tradeoff explicit; `filtered` is the honest measurement of each matcher,
  `filled` the dense final product.
* **RAFT-Stereo achieves the best dense MAE at ~1/10 of the resolution**, at
  100% coverage — but the classical `filtered` stages keep far better sub-pixel
  disparity precision (bad>2px 4.9% for Census vs 38.7% for upsampled RAFT).
* **DUSt3R** is scale-ambiguous (rescaled via the baseline, residual scale
  ~1.03) and structurally coarser (median AE 0.23 m); it has no disparity
  output, so the pixel columns are empty.
* Classical coverage tops out around ~93% because the left/right image bands
  are never seen by the second camera; no fill can recover them.

### Results — `facade` (DSC_0336/0337, baseline 0.7839 m, GT-valid 4,790,076 px)

| Method (stage) | native res | coverage | MAE (m) | median AE (m) | bad > 2 px |
|---|---|---:|---:|---:|---:|
| OpenCV SGBM+WLS (final) | 6204×4132 | 73.5% | **0.0545** | 0.0398 | **3.5%** |
| NCC raw | 6204×4132 | 46.7% | 0.0923 | 0.0518 | 7.9% |
| NCC filtered | 6204×4132 | 45.4% | 0.0754 | 0.0494 | 6.3% |
| NCC filled | 6204×4132 | 82.9% | 0.2675 | 0.0652 | 21.4% |
| SSD raw | 6204×4132 | 48.1% | 0.0954 | 0.0526 | 9.9% |
| SSD filtered | 6204×4132 | 46.9% | 0.0776 | 0.0503 | 8.4% |
| SSD filled | 6204×4132 | 87.6% | 0.3279 | 0.0670 | 23.1% |
| Census raw | 6204×4132 | 57.6% | 0.0667 | 0.0465 | 6.4% |
| Census filtered | 6204×4132 | 56.9% | 0.0603 | 0.0458 | 5.8% |
| Census filled | 6204×4132 | 84.0% | 0.2370 | 0.0575 | 18.7% |
| DUSt3R | 512×336 | 98.1% | 0.9576 | 0.9110 | — |
| RAFT-Stereo | 640×416 | **100.0%** | 0.1613 | 0.0800 | 26.7% |

The second scene changes the picture in instructive ways:

* **SSD recovers completely** (48.1% raw coverage vs 11.2% on `delivery_area`):
  the facade pair is texture-rich with consistent exposure, so SSD's missing
  photometric normalization no longer hurts. Its `delivery_area` collapse is an
  illumination problem, not an implementation problem — exactly the tradeoff
  the cost-metric comparison is meant to expose.
* **Tuned OpenCV SGBM+WLS leads on this scene** (73.5% / 0.0545 m) — semi-global
  optimization thrives on the repetitive but high-texture facade, while our
  local matcher family stays purely local. Census remains the best custom cost.
* **DUSt3R degrades sharply** (median AE 0.91 m, residual scale 0.95): its
  metric scale comes from matching the predicted camera baseline to the
  physical one, which is fragile — a ~5% baseline error becomes a ~1 m depth
  bias at this scene's ~18 m median depth. RAFT-Stereo, which predicts
  disparity directly, stays solid (100% / 0.161 m).
* Absolute errors are larger for everyone than on `delivery_area` — depth error
  grows quadratically with distance at fixed disparity precision, and this
  scene's median depth is ~18 m vs ~7 m.

## Dense matching backends

The dense stage is split behind the `IDenseStereoMatcher` interface, so the OpenCV
matcher and our own custom matcher are drop-in interchangeable — everything
downstream only depends on the resulting `DenseMatchingResult`. Select with one
flag, `DenseStereoConfig::backend`:

* `DenseStereoBackend::OpenCv` (`DenseStereoMatcher`) — OpenCV StereoBM/StereoSGBM
  with the optional WLS refinement described above. Its knobs live in
  `DenseStereoConfig::openCv` (`OpenCvStereoConfig`).
* `DenseStereoBackend::Custom` (`CustomDenseMatcher`) — **our own** hierarchical
  coarse-to-fine matcher. It builds an image pyramid (from
  `custom.coarsestDownscale` ≈ 0.0625 up to `custom.finalDownscale` = 0.5) and
  does a **full, width-derived disparity search** on the tiny coarsest level (no
  hand-tuned `numDisparities`); each finer level then warps the target image by
  the upsampled prior disparity (`cv::remap`) and searches only a small
  `±custom.residualRadius` residual around it. It builds the cost volume,
  winner-take-all disparity, parabola subpixel refinement, a full-pyramid
  left-right consistency check, and per-metric confidence gates itself; it does
  **not** use any OpenCV stereo class. OpenCV is used only for box-filter window
  sums, resize/remap and image arithmetic as numerical primitives.

The matching **cost is swappable** (`DenseStereoConfig::customCost.metric`,
dispatched by `RunCostSweep`); the surrounding pyramid/consistency/fill pipeline
is identical for all three, so the cost metrics can be ablated cleanly:

* **NCC** (`NccCostConfig`: `windowSize` 9, `minCorrelation` 0.6) — normalized
  cross-correlation; invariant to affine brightness/contrast changes between
  the two cameras.
* **SSD** (`SsdCostConfig`: `windowSize` 9, `maxCost` 0.006, `minUniqueness`
  0.05) — normalized mean squared intensity difference with a max-cost gate and
  a uniqueness check (winner must beat the second-best candidate by a margin).
  No photometric normalization, so it is the most exposure-sensitive of the
  three — visible in its results.
* **Census** (`CensusCostConfig`: `descriptorRadius` 2 → 5×5/24-bit descriptor,
  `aggregationWindowSize` 9, `maxCost`, `minUniqueness` 0.05) — census
  transform + aggregated Hamming distance; depends only on local intensity
  *ranks*, making it the most illumination-robust.

(One transparency note for the numbers: NCC uses its correlation gate but no
uniqueness check, while SSD/Census have one — the per-metric confidence gating
is not identical.)

`CreateDenseMatcher(config)` returns the right implementation. `main` runs all
four configurations (OpenCV + the three custom costs) so the maps can be
compared directly.

Deriving the disparity range from image geometry means the custom matcher needs
no `numDisparities` bound: the coarsest level searches `round(coarsest_width ·
custom.maxDisparityFraction)` (default 1/3 of the width) and the pyramid
refines from there, so the final disparity is produced at
`custom.finalDownscale` and then upscaled/rescaled to full resolution, keeping
`DepthReconstructor` unchanged. After matching, a valid-aware median
(`custom.medianKernel`, over valid neighbours only) and our own
connected-component speckle filter (`custom.speckleMinArea`,
`custom.speckleTolerance`) clean the map at working resolution — producing the
`filtered` stage. All shared knobs live in `DenseStereoConfig::custom`
(`CustomStereoConfig`); see the header comments in
`include/DenseStereoMatcher.h` for the full list.

### Occlusion-aware densification

The `filtered` map is accurate but sparse (e.g. ~36% of the GT region for NCC on
`delivery_area`). A densification pass then fills the holes without disturbing
the genuinely matched pixels, producing the `filled` stage:

1. **`FillHolesDirectional`** (`custom.fillHoles`) — an SGM-style directional fill. For
   each hole pixel it scans the 8 compass directions for the nearest valid disparity;
   when the failure reason is an LR-occlusion it takes the **second-smallest** of the
   directional candidates (Hirschmüller's occlusion rule — occlusions belong to the
   background, so the smallest/nearest value is skipped), otherwise the **median**. A
   pixel is only filled when a valid disparity is reachable in at least
   `custom.fillMinDirections = 5` of the 8 directions: the left no-correspondence band
   (~20% of the GT region) has exactly 5 directions available, so 5 fills it while 6
   would leave it (and other genuinely-unobserved strips) permanently blank.
2. **`GuidedDiffusionRefine`** (`custom.guidedFillIterations`) — **ships disabled (0)**.
   It relaxed mismatch-type fills toward the surrounding measurements via a guide-weighted
   normalized convolution, but in textureless holes the guide has no image edges, so it
   degenerated to plain averaging and smeared depth across discontinuities that carry no
   edge (MAE 0.220 → 0.362 m on `delivery_area`). Kept as an ablation, off by default.
3. **`WeightedMedianRefine`** (`custom.weightedMedianRadius = 9`,
   `custom.weightedMedianIterations = 2`) — an edge-aware weighted median applied to the
   **filled pixels only**, guided by the rectified left image so the fill follows image
   edges instead of crossing them. Two passes at radius 9 propagate edges into the filled
   regions; matched pixels are left untouched.

To keep the interpolated fills honest, every custom run exports all three stages
separately (disparity/depth/point cloud/mesh per stage — see [Outputs](#outputs))
plus the per-pixel pre-fill `failureReason` map (`reason_<tag>.png`,
`CustomMatchReason` codes: matched / never-matched / border / LR-rejected /
speckle), and `evaluate_depth.py` reports each stage as its own row.

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
  dense `filled`-stage meshes around ~220 MB (the fill-free `filtered`-stage
  meshes are much smaller) instead of the multi-GB an ASCII full-grid export
  would produce. Poisson meshing can be added later.

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
* **Custom matcher** (`DenseStereoBackend::Custom`): the hierarchical
  coarse-to-fine search recovers the same scene structure / near-far gradient as the
  OpenCV map and, because the coarsest level derives its disparity range from the
  image width, needs no hand-tuned search bound to reach the nearest geometry. It can
  still be noisier on textureless regions such as the floor, where window-based
  costs are unreliable regardless of the metric.
  Tighten the per-cost gates (`customCost.ncc.minCorrelation`,
  `customCost.ssd.minUniqueness`, `customCost.census.minUniqueness`) and/or
  `custom.lrConsistency`, raise the cost window sizes, or grow the valid-aware
  post-filters (`custom.medianKernel`, `custom.speckleMinArea`) to trade density
  for cleanliness; lower `custom.finalDownscale` (0.5 → 0.25) if the 25 MP
  pairs are too slow.
* The recovered translation vector currently provides direction only, so reconstruction is up to scale. Metric depth requires the known camera baseline (from `images.txt`).
* **Left/right ordering:** disparity matching assumes the second camera is to the right (`t.x < 0`). `main` now checks the recovered translation and, if `t.x > 0`, swaps the pair and recomputes the geometry so disparities are correct (on `delivery_area` this raised valid-disparity coverage from ~13% to ~20%). `DepthReconstructor` also keeps a cloud-orientation safety net.
* **Known issue to address:** `DenseStereoMatcher` calls `StereoBM::compute(right, left)` but `StereoSGBM::compute(left, right)` — the inconsistent argument order flips the disparity sign for the BM path (the SGBM path used by default is correct).
* Per the TA, the OpenCV building blocks (8-point algorithm, rectification, dense matching, depth estimation) had to be replaced with our own custom implementations — **all four are done** and active as the primary path; the OpenCV equivalents remain only as the comparison baseline.
