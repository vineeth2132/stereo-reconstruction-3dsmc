#pragma once
#include <opencv2/opencv.hpp>
#include "StereoRectifier.h"

enum class DenseStereoMethod
{
	StereoBM,
	StereoSGBM
};

struct DenseStereoConfig
{
	DenseStereoMethod method = DenseStereoMethod::StereoSGBM;

	int minDisparity = 0;
	int numDisparities = 192;
	int blockSize = 5;

	// StereoSGBM only
	int p1 = 0;
	int p2 = 0;
};

struct DenseMatchingResult
{
	cv::Mat rawDisparity;
	cv::Mat disparityVisualization;
	cv::Mat validDisparityMask;

	// for debugging
	void ShowDisparity() const;
};

class DenseStereoMatcher
{
public:
	DenseMatchingResult ComputeDisparity(const RectificationResult& rectificationResult, const DenseStereoConfig& config) const;

private:
	cv::Mat ConvertToGrayscale(const cv::Mat& image) const;
	DenseMatchingResult BuildResult(const cv::Mat& rawDisparity, int minDisparity) const;
};
