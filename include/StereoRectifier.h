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
	Eigen::Matrix3d leftRectificationRotation;
	Eigen::Matrix<double, 3, 4> leftProjectionMatrix;

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
	Eigen::Matrix3d ConvertToEigenMatrix3d(const cv::Mat& mat) const;
	Eigen::Matrix<double, 3, 4> ConvertToEigenMatrix3x4d(const cv::Mat& mat) const;
	cv::Mat ConvertToOpenCvVector(const Eigen::Vector3d& vector) const;
	cv::Mat ConvertToOpenCvMatrix(const Eigen::Matrix3d& matrix) const;
};
