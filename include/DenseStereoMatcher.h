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
	int numDisparities = 320;
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
	// only uses OpenCV for box filtering and image arithmetic as numerical
	// primitives. It is O(numDisparities) box-filter passes, so it is run on a
	// downscaled rectified pair for tractable runtime, then the disparity is
	// upscaled and rescaled back to full resolution.

	CustomCostMetric customCostMetric = CustomCostMetric::NCC;

	// Working resolution as a fraction of the full rectified image (1.0 = full).
	// Smaller -> much faster, coarser disparities. 0.25 is a good default on the
	// 25 MP ETH3D pairs.
	double customDownscale = 0.25;

	// Correlation window (odd) and disparity search range, both in *downscaled*
	// pixels. customNumDisparities need not be a multiple of 16 (that is an
	// OpenCV-only constraint).
	int customWindowSize = 9;
	int customMinDisparity = 0;
	int customNumDisparities = 96;

	// Left-right consistency: re-match right->left and reject pixels whose two
	// disparities disagree by more than this (downscaled px). <= 0 disables.
	float customLrConsistency = 1.0f;

	// Reject matches whose best NCC correlation is below this (0..1) as untextured
	// / unreliable.
	float customMinCorrelation = 0.5f;

	// Parabola subpixel refinement of the winning disparity.
	bool customSubpixel = true;
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
