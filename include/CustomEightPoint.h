#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

class CustomEightPoint
{
public:
    static cv::Mat EstimateFundamental(
        const std::vector<cv::Point2f>& points1,
        const std::vector<cv::Point2f>& points2
    );

private:
    static void NormalizePoints(
        const std::vector<cv::Point2f>& points,
        std::vector<cv::Point2f>& normalizedPoints,
        cv::Mat& T
    );
};