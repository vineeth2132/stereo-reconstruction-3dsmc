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

namespace
{
    std::string ToLower(std::string s)
    {
        std::transform(s.begin(), s.end(), s.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        return s;
    }

    std::string ImageKey(const std::string& imageName)
    {
        // Convert paths such as:
        //   DSC_0688.jpg
        //   dslr_images_undistorted/DSC_0688.JPG
        // into a common key:
        //   dsc_0688
        return ToLower(std::filesystem::path(imageName).stem().string());
    }

    struct ColmapImagePose
    {
        Eigen::Quaterniond q;
        Eigen::Vector3d t;

        Eigen::Vector3d CameraCenter() const
        {
            // COLMAP stores world-to-camera pose:
            //   x_cam = R * x_world + t
            //
            // The camera center in world coordinates is:
            //   C = -R^T * t
            Eigen::Matrix3d R = q.normalized().toRotationMatrix();
            return -R.transpose() * t;
        }
    };
}

float DataLoader::LoadMetricBaselineFromImagesTxt(const std::string& imagesTxtRelativePath, const std::string& leftImgName, const std::string& rightImgName)
{
    const std::filesystem::path imagesTxtPath = dataRootDirectory / imagesTxtRelativePath;
    // Use the DataLoader root path and append the relative images.txt path.

    std::ifstream file(imagesTxtPath);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open images.txt: " + imagesTxtPath.string());
    }

    const std::string leftKey = ImageKey(leftImgName);
    const std::string rightKey = ImageKey(rightImgName);

    bool foundLeft = false;
    bool foundRight = false;

    ColmapImagePose leftPose;
    ColmapImagePose rightPose;

    std::string line;
    while (std::getline(file, line))
    {
        if (line.empty() || line[0] == '#')
            continue;

        std::istringstream iss(line);

        int imageId;
        double qw, qx, qy, qz;
        double tx, ty, tz;
        int cameraId;
        std::string imagePath;

        if (!(iss >> imageId >> qw >> qx >> qy >> qz >> tx >> ty >> tz >> cameraId >> imagePath))
            continue;

        const std::string currentKey = ImageKey(imagePath);

        if (currentKey == leftKey)
        {
            leftPose.q = Eigen::Quaterniond(qw, qx, qy, qz);
            leftPose.t = Eigen::Vector3d(tx, ty, tz);
            foundLeft = true;
        }

        if (currentKey == rightKey)
        {
            rightPose.q = Eigen::Quaterniond(qw, qx, qy, qz);
            rightPose.t = Eigen::Vector3d(tx, ty, tz);
            foundRight = true;
        }

        // In COLMAP's images.txt format, each pose line is followed by a POINTS2D line.
        // Skip that line because it is not needed for baseline computation.
        std::getline(file, line);

        if (foundLeft && foundRight)
            break;
    }

    if (!foundLeft)
        throw std::runtime_error("Left image pose not found in images.txt: " + leftImgName);

    if (!foundRight)
        throw std::runtime_error("Right image pose not found in images.txt: " + rightImgName);

    const Eigen::Vector3d C_left = leftPose.CameraCenter();
    const Eigen::Vector3d C_right = rightPose.CameraCenter();

    const double baseline = (C_right - C_left).norm();

    return static_cast<float>(baseline);
}

cv::Mat DataLoader::LoadGroundTruthDepthMap(const std::string& depthRelativePath, int width, int height)
{
    const std::filesystem::path depthPath = dataRootDirectory / depthRelativePath;

    std::ifstream file(depthPath, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open ground truth depth file: " + depthPath.string());
    }

    const size_t expectedElementCount = static_cast<size_t>(width) * static_cast<size_t>(height);
    const size_t expectedByteCount = expectedElementCount * sizeof(float);

    file.seekg(0, std::ios::end);
    const size_t actualByteCount = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    if (actualByteCount != expectedByteCount)
    {
        throw std::runtime_error(
            "Ground truth depth size mismatch. Expected " +
            std::to_string(expectedByteCount) + " bytes, got " +
            std::to_string(actualByteCount) + " bytes."
        );
    }

    std::vector<float> depthData(expectedElementCount);

    file.read(
        reinterpret_cast<char*>(depthData.data()),
        static_cast<std::streamsize>(expectedByteCount)
    );

    if (!file)
    {
        throw std::runtime_error("Failed to read ground truth depth file: " + depthPath.string());
    }

    for (float& depthValue : depthData)
    {
        // ETH3D invalid depth values are stored as infinity.
        // Also treat non-positive values as invalid.
        if (!std::isfinite(depthValue) || depthValue <= 0.0f)
        {
            depthValue = std::numeric_limits<float>::quiet_NaN();
        }
    }

    cv::Mat depthMap(height, width, CV_32FC1, depthData.data());

    // Clone because depthData will be destroyed when this function returns.
    return depthMap.clone();
}

RawCameraCalibration DataLoader::LoadRawCameraCalibration(const std::filesystem::path& cameraFileRelativePath)
{
    const std::filesystem::path cameraFilePath = dataRootDirectory / cameraFileRelativePath;

    std::ifstream file(cameraFilePath);

    if (!file.is_open())
    {
        throw std::runtime_error(
            "Failed to open raw camera calibration file: " +
            cameraFilePath.string());
    }

    std::string line;

    while (std::getline(file, line))
    {
        if (line.empty() || line.front() == '#')
        {
            continue;
        }

        std::istringstream stream(line);

        int cameraId;
        std::string cameraModel;

        RawCameraCalibration calibration{};

        stream >>
            cameraId >>
            cameraModel >>
            calibration.imageWidth >>
            calibration.imageHeight >>
            calibration.fx >>
            calibration.fy >>
            calibration.cx >>
            calibration.cy >>
            calibration.k1 >>
            calibration.k2 >>
            calibration.p1 >>
            calibration.p2 >>
            calibration.k3 >>
            calibration.k4 >>
            calibration.sx1 >>
            calibration.sy1;

        if (!stream)
        {
            throw std::runtime_error(
                "Invalid raw camera calibration format: " +
                cameraFilePath.string());
        }

        if (cameraModel != "THIN_PRISM_FISHEYE")
        {
            throw std::runtime_error(
                "Unsupported raw camera model: " + cameraModel);
        }

        std::cout << "Raw camera calibration loaded successfully." << std::endl;

        return calibration;
    }

    throw std::runtime_error(
        "No raw camera calibration found in file: " +
        cameraFilePath.string());
}
