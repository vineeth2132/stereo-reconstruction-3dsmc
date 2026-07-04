#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

struct CustomRansacResult
{
    cv::Mat fundamentalMatrix;
    cv::Mat inlierMask;
    int inlierCount = 0;
    double meanSampsonError = 0.0;
};

class CustomEightPoint
{
public:
    static cv::Mat EstimateFundamental(
        const std::vector<cv::Point2f>& points1,
        const std::vector<cv::Point2f>& points2
    );

    static CustomRansacResult EstimateFundamentalRansac(
        const std::vector<cv::Point2f>& points1,
        const std::vector<cv::Point2f>& points2,
        int iterations = 2000,
        double sampsonThreshold = 1.0
    );

private:
    static void NormalizePoints(
        const std::vector<cv::Point2f>& points,
        std::vector<cv::Point2f>& normalizedPoints,
        cv::Mat& T
    );

    static double ComputeSampsonError(
        const cv::Mat& F,
        const cv::Point2f& point1,
        const cv::Point2f& point2
    );
};