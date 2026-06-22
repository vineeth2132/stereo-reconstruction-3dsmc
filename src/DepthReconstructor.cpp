#include "DepthReconstructor.h"

#include <array>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace
{
	bool IsFinitePoint(const cv::Vec3f& point)
	{
		return std::isfinite(point[0]) && std::isfinite(point[1]) && std::isfinite(point[2]);
	}
}

int ReconstructionResult::ValidPointCount() const
{
	return cv::countNonZero(validMask);
}

void ReconstructionResult::PrintStats() const
{
	if (points3D.empty()) { throw std::runtime_error("Cannot print stats for an empty reconstruction."); }

	const int totalPixels = points3D.rows * points3D.cols;
	const int validPixels = ValidPointCount();

	float minZ = std::numeric_limits<float>::max();
	float maxZ = std::numeric_limits<float>::lowest();
	double sumZ = 0.0;

	for (int row = 0; row < points3D.rows; ++row)
	{
		for (int col = 0; col < points3D.cols; ++col)
		{
			if (validMask.at<unsigned char>(row, col) == 0) { continue; }

			const float z = points3D.at<cv::Vec3f>(row, col)[2];
			minZ = std::min(minZ, z);
			maxZ = std::max(maxZ, z);
			sumZ += z;
		}
	}

	const double fillRatio = totalPixels == 0 ? 0.0 : static_cast<double>(validPixels) / static_cast<double>(totalPixels);

	std::cout << "Reconstruction: " << validPixels << " / " << totalPixels << " valid points (" << fillRatio * 100.0 << "%)\n";
	if (validPixels > 0)
	{
		std::cout << "Depth (Z) range: [" << minZ << ", " << maxZ << "], mean: " << sumZ / validPixels << '\n';
	}
}

void ReconstructionResult::WriteDepthMapTiff(const std::filesystem::path& outputPath) const
{
	if (points3D.empty()) { throw std::runtime_error("Cannot write depth map from empty reconstruction."); }
	if (validMask.empty()) { throw std::runtime_error("Cannot write depth map without valid mask."); }

	cv::Mat depthMap(points3D.size(), CV_32F, cv::Scalar(-1.0f));

	for (int row = 0; row < points3D.rows; ++row)
	{
		for (int col = 0; col < points3D.cols; ++col)
		{
			if (validMask.at<unsigned char>(row, col) == 0) { continue; }

			const cv::Vec3f point = points3D.at<cv::Vec3f>(row, col);
			depthMap.at<float>(row, col) = point[2];
		}
	}

	cv::imwrite((outputPath / "depth_float.tiff").string(), depthMap);
	cv::imwrite((outputPath / "valid_depth_mask.tiff").string(), validMask);

	std::cout << "Depth map written to " << outputPath.string() << std::endl;
}

cv::Mat DepthReconstructor::ConvertQToOpenCv(const Eigen::Matrix4d& reprojectionMatrixQ) const
{
	cv::Mat result(4, 4, CV_64F);

	for (int row = 0; row < 4; ++row)
	{
		for (int column = 0; column < 4; ++column)
		{
			result.at<double>(row, column) = reprojectionMatrixQ(row, column);
		}
	}

	return result;
}

ReconstructionResult DepthReconstructor::Reconstruct(const RectificationResult& rectificationResult, const DenseMatchingResult& denseResult, const DepthReconstructionConfig& config) const
{
	std::cout << "Depth reconstruction started." << std::endl;

	if (denseResult.rawDisparity.empty()) { throw std::runtime_error("Cannot reconstruct from an empty disparity map."); }
	if (rectificationResult.rectifiedLeftImage.empty()) { throw std::runtime_error("Cannot reconstruct without the rectified left image for color."); }
	if (denseResult.rawDisparity.size() != rectificationResult.rectifiedLeftImage.size())
	{
		throw std::runtime_error("Disparity map and rectified left image must have equal dimensions.");
	}

	const cv::Mat reprojectionMatrixQ = ConvertQToOpenCv(rectificationResult.reprojectionMatrixQ);

	cv::Mat points3D;
	cv::reprojectImageTo3D(denseResult.rawDisparity, points3D, reprojectionMatrixQ, true, CV_32F);

	/*
		The Q matrix was built from a unit-length translation, so the result is
		only known up to scale. Multiply by the real baseline to recover metric
		depth when it is available.
	*/
	if (config.metricBaseline != 1.0)
	{
		points3D *= config.metricBaseline;
	}

	/*
		Build a base mask of geometrically usable pixels (valid disparity, finite,
		not the missing-value sentinel). The reconstruction is only up to scale and
		its sign depends on the left/right ordering and chirality, so depth can come
		out negative. We measure the dominant sign and flip the cloud so that depth
		is positive (points in front of the camera) before applying the depth gate.
	*/
	cv::Mat baseMask = cv::Mat::zeros(points3D.size(), CV_8U);
	int dispValidCount = 0;
	int finiteCount = 0;
	double sumZ = 0.0;

	for (int row = 0; row < points3D.rows; ++row)
	{
		for (int col = 0; col < points3D.cols; ++col)
		{
			if (denseResult.validDisparityMask.at<unsigned char>(row, col) == 0) { continue; }
			++dispValidCount;

			const cv::Vec3f point = points3D.at<cv::Vec3f>(row, col);
			if (!IsFinitePoint(point)) { continue; }
			if (std::abs(point[2]) >= 9999.0f) { continue; } // reprojectImageTo3D missing-value sentinel
			++finiteCount;

			baseMask.at<unsigned char>(row, col) = 255;
			sumZ += point[2];
		}
	}

	const bool flipSign = finiteCount > 0 && (sumZ / finiteCount) < 0.0;
	if (flipSign)
	{
		points3D *= -1.0f; // reflect through the origin so depth becomes positive
	}

	ReconstructionResult result;
	result.points3D = points3D;
	result.colors = rectificationResult.rectifiedLeftImage; // BGR, already aligned to the disparity map
	result.validMask = cv::Mat::zeros(points3D.size(), CV_8U);

	for (int row = 0; row < points3D.rows; ++row)
	{
		for (int col = 0; col < points3D.cols; ++col)
		{
			if (baseMask.at<unsigned char>(row, col) == 0) { continue; }

			const float z = points3D.at<cv::Vec3f>(row, col)[2];
			if (z <= config.minDepth) { continue; }
			if (config.maxDepth > 0.0f && z > config.maxDepth) { continue; }

			result.validMask.at<unsigned char>(row, col) = 255;
		}
	}

	std::cout << "Depth reconstruction diagnostics: validDisparity=" << dispValidCount
		<< ", finite=" << finiteCount << ", flippedSign=" << (flipSign ? "yes" : "no")
		<< ", kept=" << result.ValidPointCount() << std::endl;
	std::cout << "Depth reconstruction finished successfully." << std::endl;

	return result;
}

void ReconstructionResult::WritePointCloudPly(const std::filesystem::path& outputPath) const
{
	if (points3D.empty()) { throw std::runtime_error("Cannot write an empty point cloud."); }

	std::filesystem::create_directories(outputPath.parent_path());

	std::ofstream file(outputPath);
	if (!file.is_open()) { throw std::runtime_error("Failed to open output file: " + outputPath.string()); }

	const int vertexCount = ValidPointCount();

	file << "ply\n";
	file << "format ascii 1.0\n";
	file << "element vertex " << vertexCount << "\n";
	file << "property float x\nproperty float y\nproperty float z\n";
	file << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
	file << "end_header\n";

	for (int row = 0; row < points3D.rows; ++row)
	{
		for (int col = 0; col < points3D.cols; ++col)
		{
			if (validMask.at<unsigned char>(row, col) == 0) { continue; }

			const cv::Vec3f point = points3D.at<cv::Vec3f>(row, col);
			const cv::Vec3b color = colors.at<cv::Vec3b>(row, col); // BGR

			file << point[0] << ' ' << point[1] << ' ' << point[2] << ' '
				<< static_cast<int>(color[2]) << ' ' << static_cast<int>(color[1]) << ' ' << static_cast<int>(color[0]) << '\n';
		}
	}

	std::cout << "Point cloud written to " << outputPath.string() << " (" << vertexCount << " points)." << std::endl;
}

void ReconstructionResult::WriteMeshPly(const std::filesystem::path& outputPath, float maxEdgeDepthDiff) const
{
	if (points3D.empty()) { throw std::runtime_error("Cannot mesh an empty reconstruction."); }

	// Assign a contiguous vertex index to every valid pixel.
	cv::Mat vertexIndex(points3D.size(), CV_32S, cv::Scalar(-1));
	int nextIndex = 0;

	double sumZ = 0.0;
	for (int row = 0; row < points3D.rows; ++row)
	{
		for (int col = 0; col < points3D.cols; ++col)
		{
			if (validMask.at<unsigned char>(row, col) == 0) { continue; }
			vertexIndex.at<int>(row, col) = nextIndex++;
			sumZ += points3D.at<cv::Vec3f>(row, col)[2];
		}
	}

	if (nextIndex == 0) { throw std::runtime_error("No valid vertices to mesh."); }

	// Default discontinuity threshold: a small fraction of the mean depth.
	float depthThreshold = maxEdgeDepthDiff;
	if (depthThreshold <= 0.0f)
	{
		const float meanZ = static_cast<float>(sumZ / nextIndex);
		depthThreshold = 0.02f * meanZ;
	}

	const auto isQuadConnected = [&](int r, int c) -> bool
	{
		const float z00 = points3D.at<cv::Vec3f>(r, c)[2];
		const float z01 = points3D.at<cv::Vec3f>(r, c + 1)[2];
		const float z10 = points3D.at<cv::Vec3f>(r + 1, c)[2];
		const float z11 = points3D.at<cv::Vec3f>(r + 1, c + 1)[2];

		const float maxZ = std::max(std::max(z00, z01), std::max(z10, z11));
		const float minZ = std::min(std::min(z00, z01), std::min(z10, z11));

		return (maxZ - minZ) <= depthThreshold;
	};

	std::vector<std::array<int, 3>> faces;

	for (int row = 0; row < points3D.rows - 1; ++row)
	{
		for (int col = 0; col < points3D.cols - 1; ++col)
		{
			const int i00 = vertexIndex.at<int>(row, col);
			const int i01 = vertexIndex.at<int>(row, col + 1);
			const int i10 = vertexIndex.at<int>(row + 1, col);
			const int i11 = vertexIndex.at<int>(row + 1, col + 1);

			if (i00 < 0 || i01 < 0 || i10 < 0 || i11 < 0) { continue; }
			if (!isQuadConnected(row, col)) { continue; }

			faces.push_back({ i00, i10, i11 });
			faces.push_back({ i00, i11, i01 });
		}
	}

	std::filesystem::create_directories(outputPath.parent_path());

	std::ofstream file(outputPath);
	if (!file.is_open()) { throw std::runtime_error("Failed to open output file: " + outputPath.string()); }

	file << "ply\n";
	file << "format ascii 1.0\n";
	file << "element vertex " << nextIndex << "\n";
	file << "property float x\nproperty float y\nproperty float z\n";
	file << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
	file << "element face " << faces.size() << "\n";
	file << "property list uchar int vertex_indices\n";
	file << "end_header\n";

	for (int row = 0; row < points3D.rows; ++row)
	{
		for (int col = 0; col < points3D.cols; ++col)
		{
			if (validMask.at<unsigned char>(row, col) == 0) { continue; }

			const cv::Vec3f point = points3D.at<cv::Vec3f>(row, col);
			const cv::Vec3b color = colors.at<cv::Vec3b>(row, col); // BGR

			file << point[0] << ' ' << point[1] << ' ' << point[2] << ' '
				<< static_cast<int>(color[2]) << ' ' << static_cast<int>(color[1]) << ' ' << static_cast<int>(color[0]) << '\n';
		}
	}

	for (const std::array<int, 3>& face : faces)
	{
		file << "3 " << face[0] << ' ' << face[1] << ' ' << face[2] << '\n';
	}

	std::cout << "Mesh written to " << outputPath.string() << " (" << nextIndex << " vertices, " << faces.size() << " faces)." << std::endl;
}
