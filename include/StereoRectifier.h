#pragma once
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "DataLoader.h"
#include "StereoGeometryEstimator.h"

struct RectificationResult
{
	cv::Mat rectifiedLeftImage;
	cv::Mat rectifiedRightImage;

	Eigen::Matrix4d reprojectionMatrixQ;

	// for debugging
	void ShowRectifiedImages() const;
};

class StereoRectifier
{
public:
	RectificationResult Rectify(const StereoImagePair& imagePair, const CameraIntrinsics& intrinsics, const StereoGeometry& geometry) const;

private:
	cv::Mat CreateOpenCvCameraMatrix(const CameraIntrinsics& intrinsics) const;
	cv::Mat CreateZeroDistortionCoefficients() const;
	Eigen::Matrix4d ConvertToEigenMatrix4d(const cv::Mat& matrix) const;
	cv::Mat ConvertToOpenCvVector(const Eigen::Vector3d& vector) const;
	cv::Mat ConvertToOpenCvMatrix(const Eigen::Matrix3d& matrix) const;
};
