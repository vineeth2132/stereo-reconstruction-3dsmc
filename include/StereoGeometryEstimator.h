#pragma once
#include <Eigen/Dense>
#include <vector>
#include <opencv2/opencv.hpp>

#include "DataLoader.h"
#include "SparseFeatureMatcher.h"

struct StereoGeometry
{
    Eigen::Matrix3d fundamentalMatrix;
    Eigen::Matrix3d essentialMatrix;

    Eigen::Matrix3d rotation;
    Eigen::Vector3d translation;

    std::vector<unsigned char> inlierMask;

    // for debugging
    void PrintData() const;
};

class StereoGeometryEstimator
{
public:
    StereoGeometry EstimateGeometry(const SparseMatchingResult& matchingResult, const CameraIntrinsics& intrinsics) const;

private:
    cv::Mat CreateOpenCvCameraMatrix(const CameraIntrinsics& intrinsics) const;
    Eigen::Matrix3d ConvertToEigenMatrix3d(const cv::Mat& matrix) const;
    Eigen::Vector3d ConvertToEigenVector3d(const cv::Mat& vector) const;
};