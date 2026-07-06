#pragma once
#include <vector>

#include <opencv2/opencv.hpp>

#include "DenseStereoMatcher.h"
#include "StereoRectifier.h"

/*
	Our own dense stereo matcher: a hierarchical coarse-to-fine
	normalized-cross-correlation (NCC) block matcher with subpixel refinement, a
	left-right consistency check and our own median/speckle post-filtering. It
	does NOT use any OpenCV stereo class (StereoBM/StereoSGBM/ximgproc/
	filterSpeckles); OpenCV is used only for image arithmetic, box filtering,
	resize and remap as numerical primitives, so the matching algorithm itself
	(pyramid, cost volume, winner-take-all, subpixel, warping, LR check, median,
	speckle removal) is implemented here.

	The coarsest pyramid level carries an unconstrained full disparity search
	(range derived from the image width, not a hand-tuned bound); every finer
	level only searches a small residual around the upsampled prior, so the
	overall cost stays close to a single fine-resolution pass.

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
		Build the image pyramid for one view: each entry is the full-resolution
		float image resized (INTER_AREA) to the corresponding working scale. The
		scales are ordered coarsest -> finest to match the processing order.
	*/
	std::vector<cv::Mat> BuildPyramid(const cv::Mat& fullFloat, const std::vector<double>& scales) const;

	/*
		Run the full coarse-to-fine NCC pyramid for a single matching direction and
		return the finest-level disparity (in finest working-resolution pixels;
		pixels that never found a confident match hold the invalid sentinel).
		leftToRight selects the disparity sign convention so the same routine serves
		the LR-consistency reverse pass.
	*/
	cv::Mat MatchDirectionPyramid(const std::vector<cv::Mat>& referencePyramid, const std::vector<cv::Mat>& targetPyramid, const std::vector<double>& scales, const DenseStereoConfig& config, bool leftToRight) const;
};
