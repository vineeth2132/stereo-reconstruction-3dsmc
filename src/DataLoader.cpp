#include "DataLoader.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

void StereoImagePair::ShowResized(const float resizeRatio) const
{
    cv::Mat resizedLeft;
    cv::Mat resizedRight;

    cv::resize(leftImage, resizedLeft, cv::Size(), resizeRatio, resizeRatio);
    cv::resize(rightImage, resizedRight, cv::Size(), resizeRatio, resizeRatio);

    cv::imshow("Left Image", resizedLeft);
    cv::waitKey(0);

    cv::imshow("Right Image", resizedRight);
    cv::waitKey(0);

    cv::destroyAllWindows();
}

void CameraIntrinsics::Print() const
{
    std::cout << "width : " << imageWidth << '\n';
    std::cout << "height : " << imageHeight << '\n';
    std::cout << "fx : " << fx << '\n';
    std::cout << "fy : " << fy << '\n';
    std::cout << "cx : " << cx << '\n';
    std::cout << "cy : " << cy << '\n';
}

DataLoader::DataLoader(std::filesystem::path dataRootDir_) : dataRootDirectory(std::move(dataRootDir_))
{ }

StereoImagePair DataLoader::LoadStereoPair(const std::filesystem::path& leftImgRelativePath, const std::filesystem::path& rightImgRelativePath)
{
	const std::filesystem::path leftImagePath = dataRootDirectory / leftImgRelativePath;
	const std::filesystem::path rightImagePath = dataRootDirectory / rightImgRelativePath;

	StereoImagePair imagePair;
	imagePair.leftImgPath = leftImagePath;
	imagePair.rightImgPath = rightImagePath;
	imagePair.leftImage = LoadImage(imagePair.leftImgPath);
	imagePair.rightImage = LoadImage(imagePair.rightImgPath);

    std::cout << "Stereo image pair loaded successfully." << std::endl;

	return imagePair;
}

cv::Mat DataLoader::LoadImage(const std::filesystem::path& imagePath)
{
	cv::Mat image = cv::imread(imagePath.string(), cv::IMREAD_COLOR);

	if (image.empty()) { throw std::runtime_error("Failed to load image: " + imagePath.string()); }

	return image;
}

CameraIntrinsics DataLoader::LoadCameraIntrinsics(const std::filesystem::path& cameraFileRelativePath)
{
	const std::filesystem::path cameraFilePath = dataRootDirectory / cameraFileRelativePath;

    std::ifstream file(cameraFilePath);

    if (!file.is_open())
    {
        throw std::runtime_error( "Failed to open camera file: " + cameraFilePath.string());
    }

    std::string line;

    while (std::getline(file, line))
    {
        if (line.empty() || line.front() == '#') { continue; }

        std::istringstream stream(line);

        int cameraId;
        std::string cameraModel;

        CameraIntrinsics intrinsics{};

        stream >>
            cameraId >>
            cameraModel >>
            intrinsics.imageWidth >>
            intrinsics.imageHeight >>
            intrinsics.fx >>
            intrinsics.fy >>
            intrinsics.cx >>
            intrinsics.cy;

        if (!stream) { throw std::runtime_error( "Invalid camera intrinsics format: " + cameraFilePath.string()); }
        if (cameraModel != "PINHOLE") { throw std::runtime_error("Unsupported camera model: " + cameraModel); }
        
        std::cout << "Camera Intrinsics loaded successfully." << std::endl;

        return intrinsics;
    }

    throw std::runtime_error(
        "No camera intrinsics found in file: " +
        cameraFilePath.string());

}

