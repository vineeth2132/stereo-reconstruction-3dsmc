#pragma once
#include <memory>
#include <opencv2/opencv.hpp>
#include "StereoRectifier.h"

/*
	Selects which dense-stereo implementation runs behind the common
	IDenseStereoMatcher interface.

	OpenCv:
		Uses OpenCV StereoBM or StereoSGBM, optionally followed by WLS filtering.

	Custom:
		Uses the custom hierarchical coarse-to-fine stereo pipeline with a
		selectable NCC, SSD or Census matching cost.
*/
enum class DenseStereoBackend
{
	OpenCv,
	Custom
};

/*
	OpenCV stereo-matching method used by the OpenCv backend.
*/
enum class DenseStereoMethod
{
	StereoBM,
	StereoSGBM
};

/*
	Matching cost used during the disparity search of the custom backend.

	The selected cost only changes how candidate disparities are evaluated.
	The remaining custom pipeline, including pyramid refinement, left-right
	consistency, filtering, hole filling and reconstruction, is shared.
*/
enum class CustomCostMetric
{
	NCC,
	SSD,
	Census
};

/*
	Configuration for the OpenCV StereoBM/StereoSGBM backend.
*/
struct OpenCvStereoConfig
{
	DenseStereoMethod method = DenseStereoMethod::StereoSGBM;

	/*
		Disparity search range.

		numDisparities must be positive and divisible by 16, as required by
		OpenCV's stereo matchers.
	*/
	int minDisparity = 0;
	int numDisparities = 704;

	/*
		Matching block size. Must be a positive odd number.
	*/
	int blockSize = 5;

	/*
		StereoSGBM smoothness penalties.

		A value of 0 causes the implementation to derive them from blockSize:
			P1 = 8  * blockSize^2
			P2 = 32 * blockSize^2
	*/
	int p1 = 0;
	int p2 = 0;

	/*
		StereoBM/StereoSGBM quality and consistency parameters.
	*/
	int uniquenessRatio = 15;
	int speckleWindowSize = 200;
	int speckleRange = 1;
	int disp12MaxDiff = 1;
	int preFilterCap = 31;

	/*
		Median filtering applied to the resulting disparity map.

		Must be odd and at least 3 to enable filtering.
		Set to 0 to disable it.
	*/
	int medianKernel = 3;

	/*
		Edge-aware WLS disparity filtering.

		Requires the OpenCV ximgproc module. If ximgproc is unavailable, the
		implementation falls back to the unfiltered OpenCV disparity map.
	*/
	bool useWlsFilter = true;
	double wlsLambda = 8000.0;
	double wlsSigma = 1.5;

	/*
		Minimum WLS confidence required for a disparity to remain valid.

		The confidence map is produced from the left-right disparity check.
		A value of 0 keeps all WLS-filled disparities.
	*/
	float wlsConfidenceThreshold = 0.5f;
};

/*
	Parameters used only by the NCC matching cost.
*/
struct NccCostConfig
{
	/*
		Size of the square NCC aggregation window.
		Must be a positive odd number.
	*/
	int windowSize = 9;

	/*
		Minimum accepted NCC score.
		NCC is approximately in the range [-1, 1].
		Higher values indicate a better match.
	*/
	float minCorrelation = 0.6f;
};

/*
	Parameters used only by the SSD matching cost.
*/
struct SsdCostConfig
{
	/*
		Size of the square SSD aggregation window.
		Must be a positive odd number.
	*/
	int windowSize = 9;

	/*
		Maximum accepted normalized mean SSD cost.
		The SSD implementation will later be changed so that image intensities
		are treated in the range [0, 1]. The resulting mean squared cost is
		therefore also approximately in the range [0, 1].
	*/
	float maxCost = 0.006f;

	/*
		Minimum relative separation between the best and second-best costs.
		Larger values reject ambiguous matches more aggressively.
	*/
	float minUniqueness = 0.05f;
};

/*
	Parameters used only by the Census matching cost.
*/
struct CensusCostConfig
{
	/*
		Radius of the Census descriptor neighbourhood
		radius = 1 -> 3x3 descriptor, 8 comparison bits
		radius = 2 -> 5x5 descriptor, 24 comparison bits
		The current descriptor uses 32 bits, so the radius must not exceed 2.
	*/
	int descriptorRadius = 2;

	/*
		Size of the square window used to aggregate normalized Hamming costs.
		Must be a positive odd number.
	*/
	int aggregationWindowSize = 9;

	/*
		Maximum accepted normalized Census cost.
		After normalization by the descriptor bit count, the cost is in [0, 1].
	*/
	float maxCost = 15.0f / 24.0f;

	/*
		Minimum relative separation between the best and second-best costs.
	*/
	float minUniqueness = 0.05f;
};

/*
	Cost-function-specific configuration
	Only the configuration associated with the selected metric is used.
*/
struct CustomCostConfig
{
	CustomCostMetric metric = CustomCostMetric::NCC;

	NccCostConfig ncc;
	SsdCostConfig ssd;
	CensusCostConfig census;
};

/*
	Shared configuration for the custom hierarchical coarse-to-fine matcher.

	These settings are used independently of whether NCC, SSD or Census is
	selected as the matching cost.
*/
struct CustomStereoConfig
{
	/*
		Finest working resolution as a fraction of the full rectified image.

		The final working-resolution disparity map is upscaled to full resolution
		after matching and post-processing.
	*/
	double finalDownscale = 0.5;

	/*
		Coarsest pyramid resolution as a fraction of the full image.

		The coarsest level performs the full disparity search. Finer levels search
		only a small residual disparity around the upsampled prior.
	*/
	double coarsestDownscale = 0.0625;

	/*
		Half-range of the residual disparity search at each refinement level.

		For example, a value of 4 searches residual disparities from -4 to +4.
	*/
	int residualRadius = 4;

	/*
		Maximum coarsest-level disparity as a fraction of the coarsest image width.
	*/
	double maxDisparityFraction = 1.0 / 3.0;

	/*
		Left-right consistency threshold in working-resolution pixels.

		Matches whose forward and reverse disparities differ by more than this
		value are rejected. A value less than or equal to 0 disables the check.
	*/
	float lrConsistency = 1.0f;

	/*
		Enables parabola-based subpixel refinement for cost functions that support
		it.
	*/
	bool subpixel = true;

	/*
		Valid-aware median filtering applied after matching.

		Only already-valid disparities are modified; invalid pixels are not filled.
		Set to 0 to disable it.
	*/
	int medianKernel = 3;

	/*
		Speckle filtering parameters.

		Small connected disparity regions are removed when their area is below
		speckleMinArea. Neighbouring pixels belong to the same component only when
		their disparity difference does not exceed speckleTolerance.
	*/
	int speckleMinArea = 100;
	float speckleTolerance = 1.0f;

	/*
		Enables occlusion-aware directional hole filling after the matched-only
		disparity map has been filtered.
	*/
	bool fillHoles = true;

	/*
		Minimum number of valid scan directions required before an invalid pixel
		can be filled.

		The custom fill examines the nearest valid disparity in eight directions.
	*/
	int fillMinDirections = 5;

	/*
		Guide-weighted diffusion applied to mismatch-type filled pixels.

		guidedFillIterations = 0 disables this stage.
	*/
	int guidedFillIterations = 0;
	int guidedFillRadius = 6;
	float guidedFillSigmaColor = 10.0f;

	/*
		Edge-aware weighted-median refinement applied only to filled pixels.

		Matched disparities remain unchanged.
	*/
	int weightedMedianRadius = 9;
	float weightedMedianSigmaColor = 10.0f;
	int weightedMedianIterations = 2;
};

/*
	Top-level dense-stereo configuration.

	The backend determines whether openCv or custom/customCost settings are used.
*/
struct DenseStereoConfig
{
	/*
		Backend instantiated by CreateDenseMatcher().
	*/
	DenseStereoBackend backend = DenseStereoBackend::OpenCv;

	OpenCvStereoConfig openCv;
	CustomCostConfig customCost;
	CustomStereoConfig custom;
};

/*
	Pre-fill failure-reason codes recorded by the custom matcher in
	DenseMatchingResult::failureReason.

	The map is stored as CV_8U at full resolution. The values describe why a
	pixel was invalid before directional hole filling. Filled pixels keep their
	original non-zero failure reason for diagnostics.
*/
enum CustomMatchReason : unsigned char
{
	CustomMatchReasonMatched = 0,      // Confident match kept through all gates.
	CustomMatchReasonNeverMatched = 1, // No sufficiently confident pyramid match.
	CustomMatchReasonBorder = 2,       // Incomplete window or target outside the image.
	CustomMatchReasonLrRejected = 3,   // Rejected by the left-right consistency check.
	CustomMatchReasonSpeckle = 4       // Removed by the custom speckle filter.
};

/*
	Output of a dense-stereo matcher.

	All backends return a full-resolution disparity map and validity mask.
	The custom backend additionally populates matchedMask and failureReason.
*/
struct DenseMatchingResult
{
	/*
		Full-resolution disparity map in pixel units.
	*/
	cv::Mat rawDisparity;

	/*
		Normalized 8-bit disparity image used for visualization.
	*/
	cv::Mat disparityVisualization;

	/*
		CV_8U mask indicating which rawDisparity pixels are valid.
	*/
	cv::Mat validDisparityMask;

	/*
		Custom-backend diagnostics.

		matchedMask:
			Pixels that were genuinely matched before hole filling.

		failureReason:
			Pre-fill rejection reason encoded using CustomMatchReason.
	*/
	cv::Mat matchedMask;
	cv::Mat failureReason;

	void ShowDisparity() const;
};

/*
	Common interface implemented by the OpenCV and custom dense-stereo matchers.

	The remainder of the reconstruction pipeline depends only on
	DenseMatchingResult and therefore remains identical for every backend and
	custom matching cost.
*/
class IDenseStereoMatcher
{
public:
	virtual ~IDenseStereoMatcher() = default;

	virtual DenseMatchingResult ComputeDisparity(
		const RectificationResult& rectificationResult,
		const DenseStereoConfig& config) const = 0;
};

/*
	Creates the dense-stereo implementation selected by config.backend.
*/
std::unique_ptr<IDenseStereoMatcher> CreateDenseMatcher(
	const DenseStereoConfig& config);

/*
	OpenCV StereoBM/StereoSGBM implementation of IDenseStereoMatcher.
*/
class DenseStereoMatcher : public IDenseStereoMatcher
{
public:
	DenseMatchingResult ComputeDisparity(
		const RectificationResult& rectificationResult,
		const DenseStereoConfig& config) const override;

private:
	cv::Mat ConvertToGrayscale(const cv::Mat& image) const;

	cv::Mat ComputeSgbmDisparity(
		const cv::Mat& leftImage,
		const cv::Mat& rightImage,
		const cv::Mat& leftGuide,
		const DenseStereoConfig& config,
		cv::Mat& confidenceMapOut) const;

	DenseMatchingResult BuildResult(
		const cv::Mat& rawDisparity,
		const DenseStereoConfig& config) const;
};

