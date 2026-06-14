#pragma once
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "StereoRectifier.h"
#include "DenseStereoMatcher.h"

struct DepthReconstructionConfig
{
	/*
		recoverPose() returns a unit-length translation, so the rectified
		reconstruction is only known up to scale. Set metricBaseline to the
		real camera baseline (in meters) to obtain metric depth; leave it at
		1.0 for an up-to-scale point cloud (fine for visualization).
	*/
	double metricBaseline = 1.0;

	// Depth gating in the same units as the reconstruction (after scaling).
	float minDepth = 0.0f;
	float maxDepth = 0.0f; // 0 disables the upper limit

	// Mesh: skip triangles whose vertices differ in depth by more than this
	// (suppresses stretched faces across occlusion/depth boundaries).
	float maxMeshEdgeDepthDiff = 0.0f; // 0 derives a value from the depth range
};

struct ReconstructionResult
{
	cv::Mat points3D;  // CV_32FC3, organized like the disparity map
	cv::Mat colors;    // CV_8UC3 (BGR), organized, same size as points3D
	cv::Mat validMask; // CV_8U, non-zero where points3D is usable

	int ValidPointCount() const;

	void WritePointCloudPly(const std::filesystem::path& outputPath) const;
	void WriteMeshPly(const std::filesystem::path& outputPath, float maxEdgeDepthDiff) const;

	// for debugging
	void PrintStats() const;
};

class DepthReconstructor
{
public:
	ReconstructionResult Reconstruct(const RectificationResult& rectificationResult, const DenseMatchingResult& denseResult, const DepthReconstructionConfig& config) const;

private:
	cv::Mat ConvertQToOpenCv(const Eigen::Matrix4d& reprojectionMatrixQ) const;
};
