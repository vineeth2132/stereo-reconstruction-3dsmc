#pragma once
#include <opencv2/opencv.hpp>

#include "DenseStereoMatcher.h"
#include "StereoRectifier.h"

/*
	Our own dense stereo matcher: a normalized-cross-correlation (NCC) block
	matcher with subpixel refinement and a left-right consistency check. It does
	NOT use any OpenCV stereo class (StereoBM/StereoSGBM/ximgproc); OpenCV is used
	only for image arithmetic and box filtering as numerical primitives, so the
	matching algorithm itself (cost volume, winner-take-all, subpixel, LR check)
	is implemented here.

	Drop-in replacement for DenseStereoMatcher: both implement IDenseStereoMatcher
	and return a DenseMatchingResult at full rectified resolution, so the rest of
	the pipeline (DepthReconstructor) is unchanged. Select via config.backend.
*/
class CustomDenseMatcher : public IDenseStereoMatcher
{
public:
	DenseMatchingResult ComputeDisparity(const RectificationResult& rectificationResult, const DenseStereoConfig& config) const override;

private:
	cv::Mat ConvertToGrayscale(const cv::Mat& image) const;

	/*
		Core NCC matching at the (already downscaled) working resolution. Produces
		a CV_32F disparity in downscaled pixels; pixels that fail the correlation
		or left-right test are set to the invalid sentinel.
	*/
	cv::Mat MatchNcc(const cv::Mat& leftGray, const cv::Mat& rightGray, const DenseStereoConfig& config) const;

	// One matching direction (left->right). leftToRight selects the disparity sign
	// convention so the same routine serves the LR-consistency reverse pass.
	cv::Mat MatchDirection(const cv::Mat& referenceGray, const cv::Mat& targetGray, const DenseStereoConfig& config, bool leftToRight) const;
};
