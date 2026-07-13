#pragma once
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "StereoRectifier.h"
#include "DenseStereoMatcher.h"

/*
	Selects how the disparity map is back-projected into 3D. Custom is our own
	per-pixel homogeneous back-projection (what the TA wants us to own); OpenCv
	keeps cv::reprojectImageTo3D around purely for validation/comparison.
*/
enum class DepthBackend { OpenCv, Custom };

struct DepthReconstructionConfig
{
	DepthBackend backend = DepthBackend::Custom;

	// When backend == Custom, also run cv::reprojectImageTo3D and print the max
	// absolute component difference against our result (cheap sanity check for
	// the report; the reference and ours should agree to ~1e-3).
	bool validateAgainstOpenCv = true;

	// Sample the organized point grid every exportGridStep rows/cols when writing
	// the PLY point cloud / mesh. 1 keeps every point; larger values shrink the
	// exported files (the custom matcher runs at 0.5 scale, so step 2 is lossless).
	int exportGridStep = 1;

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
		99.9 (not 98): with the dense filled disparity the far tail is real
		background, and depth is already bounded by fB / min(disparity) (~28 m on
		delivery_area), so an aggressive clip only deletes genuine geometry -- at
		98 it cut GT-verified pixels between 15.3 m and 18.3 m.
	*/
	float maxDepthPercentile = 99.9f;

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

	void WritePointCloudPly(const std::filesystem::path& outputPath, int step = 1) const;
	void WriteMeshPly(const std::filesystem::path& outputPath, float maxEdgeDepthDiff, int step = 1) const;

	// Writes depth_<tag>_float.tiff + valid_depth_<tag>_mask.tiff (or the
	// un-tagged names when tag is empty) into outputDir.
	void WriteDepthMapTiff(const std::filesystem::path& outputDir, const std::string& tag = "") const;

	// for debugging
	void PrintStats() const;
};

class DepthReconstructor
{
public:
	ReconstructionResult Reconstruct(const RectificationResult& rectificationResult, const cv::Mat& disparity, const cv::Mat& validDisparityMask, const DepthReconstructionConfig& config) const;

private:
	cv::Mat ConvertQToOpenCv(const Eigen::Matrix4d& reprojectionMatrixQ) const;
};
