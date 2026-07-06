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
	float maxDepth = 0.0f; // explicit upper limit; 0 -> derive from maxDepthPercentile

	/*
		The reconstruction is only up to scale, so an absolute maxDepth is an
		arbitrary magic number that drifts whenever the geometry/scale changes.
		When maxDepth is 0, clip everything above this percentile of the valid
		depths instead - a scene-adaptive way to drop far outliers that keeps the
		mesh viewable without hand-tuning. Set to 0 to disable clipping entirely.
	*/
	float maxDepthPercentile = 98.0f;

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

	// Writes depth_<tag>_float.tiff + valid_depth_<tag>_mask.tiff (or the
	// un-tagged names when tag is empty) into outputDir.
	void WriteDepthMapTiff(const std::filesystem::path& outputDir, const std::string& tag = "") const;

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
