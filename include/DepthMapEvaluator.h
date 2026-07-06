#pragma once

#include <opencv2/opencv.hpp>

#include "DataLoader.h"
#include "StereoRectifier.h"

class DepthMapEvaluator
{
public:
    cv::Mat RectifyGroundTruthDepth(const cv::Mat& gtDepthRaw, const RawCameraCalibration& rawCalibration, const RectificationResult& rectResult, const cv::Size& rectifiedSize);
    void Evaluate(const cv::Mat& predictedDepth, const cv::Mat& predictedValidMask, const cv::Mat& rectifiedGtDepth);
};