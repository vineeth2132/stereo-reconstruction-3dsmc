#pragma once
#include <memory>
#include <opencv2/opencv.hpp>
#include "StereoRectifier.h"

// Selects which dense-matching implementation runs behind the common
// IDenseStereoMatcher interface. OpenCv = library StereoBM/SGBM (+ optional WLS);
// Custom = our own hand-written NCC block matcher. Switch with a single flag.
enum class DenseStereoBackend
{
	OpenCv,
	Custom
};

enum class DenseStereoMethod
{
	StereoBM,
	StereoSGBM
};

// Cost metric for the custom block matcher. Only NCC is wired today; the enum
// leaves room to add SAD/SSD later without changing the interface.
enum class CustomCostMetric
{
	NCC
};

struct DenseStereoConfig
{
	// Which backend the CreateDenseMatcher factory instantiates.
	DenseStereoBackend backend = DenseStereoBackend::OpenCv;

	DenseStereoMethod method = DenseStereoMethod::StereoSGBM;

	int minDisparity = 0;
	// The true disparity range on delivery_area reaches ~690 full-res px
	// (measured by the custom matcher's width-derived coarse search and confirmed
	// against ground truth); 320 clamped all near geometry. Must be a multiple of 16.
	int numDisparities = 704;
	// 5 gives the sharpest result on delivery_area with WLS on; larger windows
	// over-smooth (note p1/p2 derive from blockSize^2, so they grow too).
	int blockSize = 5;

	// StereoSGBM smoothness penalties (0 = derive from blockSize: 8*bs^2 / 32*bs^2).
	int p1 = 0;
	int p2 = 0;

	// SGBM quality/robustness knobs (previously hard-coded in the .cpp).
	int uniquenessRatio = 15;   // higher -> reject ambiguous matches (less noise, sparser)
	int speckleWindowSize = 200; // larger -> remove bigger speckle blobs
	int speckleRange = 1;       // max disparity variation within a speckle component
	int disp12MaxDiff = 1;      // left-right consistency check (-1 disables)
	int preFilterCap = 31;

	// Post-filtering of the disparity map (applied after the matcher).
	// Median blur removes salt-and-pepper noise; kernel must be odd, 0 disables.
	int medianKernel = 3;

	// Edge-aware WLS smoothing (opencv_contrib ximgproc). Falls back to no-op
	// when the build lacks ximgproc (see HAVE_OPENCV_XIMGPROC). Hugely reduces
	// noise while preserving depth edges, guided by the rectified left image.
	bool useWlsFilter = true;
	double wlsLambda = 8000.0;  // smoothness strength
	double wlsSigma = 1.5;      // edge sensitivity (color similarity)

	/*
		WLS fills textureless/occluded regions by extrapolation. Its confidence
		map (0..1, from the left-right check) lets us drop those guessed pixels so
		coverage reflects genuinely matched geometry - important for honest
		evaluation against ground truth. 0 keeps the full (dense) fill.
	*/
	float wlsConfidenceThreshold = 0.5f;

	// --- Custom NCC block matcher (DenseStereoBackend::Custom) ------------------
	// The custom matcher is our own implementation (no OpenCV stereo classes); it
	// only uses OpenCV for box filtering, resize/remap and image arithmetic as
	// numerical primitives. It is a hierarchical coarse-to-fine NCC matcher: it
	// full-searches disparity on a tiny coarsest pyramid level, then refines that
	// estimate with a small residual search at each finer level, so the disparity
	// range is derived from the image geometry rather than a hand-tuned bound.

	CustomCostMetric customCostMetric = CustomCostMetric::NCC;

	// Finest working level as a fraction of the full rectified image. The final
	// disparity is produced here and then upscaled/rescaled to full resolution.
	// 0.5 keeps detail on the 25 MP ETH3D pairs; drop to 0.25 if too slow.
	double customFinalDownscale = 0.5;

	// Coarsest pyramid level as a fraction of full resolution. The pyramid is
	// built by repeated halving from customFinalDownscale down to (roughly) this
	// value; the coarsest level carries the unconstrained full disparity search.
	double customCoarsestDownscale = 0.0625;

	// Correlation window (odd), in working-resolution pixels. Shared across levels.
	int customWindowSize = 9;

	// Residual search half-range (+/- pixels) at each refine level: after warping
	// the target by the upsampled prior, we only look this far for the leftover
	// disparity, which keeps the per-level cost tiny.
	int customResidualRadius = 4;

	// Coarsest-level full search range = round(coarsest_width * this). Sets the
	// maximum representable disparity (in coarsest px); 1/3 of the width is a
	// generous bound that covers the near-field on the ETH3D scenes.
	double customMaxDisparityFraction = 1.0 / 3.0;

	// Left-right consistency: re-run the whole pyramid right->left and reject final
	// pixels whose two disparities disagree by more than this (working px). <= 0
	// disables.
	float customLrConsistency = 1.0f;

	// Reject matches whose best NCC correlation is below this (0..1) as untextured
	// / unreliable. Applied at every level's gate.
	// 0.6 is the measured sweet spot on delivery_area: vs 0.5 it improves both
	// accuracy AND coverage (bad matches die early in the pyramid instead of at
	// the LR check); 0.7 gains little accuracy but costs ~1.4pp coverage.
	float customMinCorrelation = 0.6f;

	// Parabola subpixel refinement of the winning (residual) disparity, applied at
	// every level.
	bool customSubpixel = true;

	// Valid-aware median filter kernel (odd) at final working resolution, our own
	// implementation over valid neighbours only. Removes salt-and-pepper noise
	// without inventing values at invalid pixels. 0 disables.
	int customMedianKernel = 3;

	// Custom speckle filter: connected components (4-connectivity, neighbours
	// joined iff both valid and within customSpeckleTolerance) smaller than this
	// many working-resolution pixels are discarded as speckle. 0 disables.
	int customSpeckleMinArea = 100;

	// Maximum |disparity difference| (working px) for two neighbours to belong to
	// the same speckle component.
	float customSpeckleTolerance = 1.0f;
};

struct DenseMatchingResult
{
	cv::Mat rawDisparity;
	cv::Mat disparityVisualization;
	cv::Mat validDisparityMask;

	// for debugging
	void ShowDisparity() const;
};

// Common interface so the OpenCV and custom matchers are drop-in interchangeable:
// the rest of the pipeline only depends on DenseMatchingResult.
class IDenseStereoMatcher
{
public:
	virtual ~IDenseStereoMatcher() = default;
	virtual DenseMatchingResult ComputeDisparity(const RectificationResult& rectificationResult, const DenseStereoConfig& config) const = 0;
};

// Picks the implementation from config.backend.
std::unique_ptr<IDenseStereoMatcher> CreateDenseMatcher(DenseStereoBackend backend);

class DenseStereoMatcher : public IDenseStereoMatcher
{
public:
	DenseMatchingResult ComputeDisparity(const RectificationResult& rectificationResult, const DenseStereoConfig& config) const override;

private:
	cv::Mat ConvertToGrayscale(const cv::Mat& image) const;
	cv::Mat ComputeSgbmDisparity(const cv::Mat& leftImage, const cv::Mat& rightImage, const cv::Mat& leftGuide, const DenseStereoConfig& config, cv::Mat& confidenceMapOut) const;
	DenseMatchingResult BuildResult(const cv::Mat& rawDisparity, const DenseStereoConfig& config) const;
};
