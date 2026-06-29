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


class DataLoader
{
public:
	DataLoader(std::filesystem::path dataRootDir_);

	StereoImagePair LoadStereoPair(const std::filesystem::path& leftImgRelativePath, const std::filesystem::path& rightImgRelativePath);
	CameraIntrinsics LoadCameraIntrinsics(const std::filesystem::path& cameraFileRelativePath);
	float LoadMetricBaselineFromImagesTxt(const std::string& imagesTxtRelativePath, const std::string& leftImgName, const std::string& rightImgName);

private:
	cv::Mat LoadImage(const std::filesystem::path& imgPath);

	std::filesystem::path dataRootDirectory;
};