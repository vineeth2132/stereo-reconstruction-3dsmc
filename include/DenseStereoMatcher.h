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
	int numDisparities = 320;
	int blockSize = 7;

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
	cv::Mat ComputeSgbmDisparity(const cv::Mat& leftImage, const cv::Mat& rightImage, const cv::Mat& leftGuide, const DenseStereoConfig& config, cv::Mat& confidenceMapOut) const;
	DenseMatchingResult BuildResult(const cv::Mat& rawDisparity, const DenseStereoConfig& config) const;
};
