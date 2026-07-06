#pragma once
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

struct StereoImagePair
{
	cv::Mat leftImage;
	cv::Mat rightImage;

	std::filesystem::path leftImgPath;
	std::filesystem::path rightImgPath;

	// for debugging
	void ShowResized(const float resizeRatio) const;
};

struct CameraIntrinsics
{
	int imageWidth;
	int imageHeight;

	double fx;
	double fy;
	double cx;
	double cy;

	// for debugging
	void Print() const;
};

struct RawCameraCalibration
{
	int imageWidth = 0;
	int imageHeight = 0;

	double fx = 0.0;
	double fy = 0.0;
	double cx = 0.0;
	double cy = 0.0;

	double k1 = 0.0;
	double k2 = 0.0;
	double p1 = 0.0;
	double p2 = 0.0;

	double k3 = 0.0;
	double k4 = 0.0;

	double sx1 = 0.0;
	double sy1 = 0.0;
};

class DataLoader
{
public:
	DataLoader(std::filesystem::path dataRootDir_);

	StereoImagePair LoadStereoPair(const std::filesystem::path& leftImgRelativePath, const std::filesystem::path& rightImgRelativePath);
	CameraIntrinsics LoadCameraIntrinsics(const std::filesystem::path& cameraFileRelativePath);
	float LoadMetricBaselineFromImagesTxt(const std::string& imagesTxtRelativePath, const std::string& leftImgName, const std::string& rightImgName);
	cv::Mat LoadGroundTruthDepthMap(const std::string& depthRelativePath, int width, int height);
	RawCameraCalibration LoadRawCameraCalibration(const std::filesystem::path& cameraFileRelativePath);

private:
	cv::Mat LoadImage(const std::filesystem::path& imgPath);

	std::filesystem::path dataRootDirectory;
};