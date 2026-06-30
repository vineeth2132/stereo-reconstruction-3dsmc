#include "StereoRectifier.h"

void RectificationResult::ShowRectifiedImages() const
{
    if (rectifiedLeftImage.empty() || rectifiedRightImage.empty())
    {
        throw std::runtime_error("Cannot visualize empty rectified images.");
    }

    cv::Mat visualization;

    cv::hconcat(rectifiedLeftImage, rectifiedRightImage, visualization);

    const int lineSpacing = std::max(visualization.rows / 20, 1);

    for (int y = 0; y < visualization.rows; y += lineSpacing)
    {
        cv::line(visualization, cv::Point(0, y), cv::Point(visualization.cols, y), cv::Scalar(0, 0, 255), 2);
    }

    cv::line(visualization, cv::Point(rectifiedLeftImage.cols, 0), cv::Point(rectifiedLeftImage.cols, visualization.rows), cv::Scalar(0, 255, 0), 3);

    cv::namedWindow("Rectified Stereo Pair", cv::WINDOW_NORMAL);
    cv::resizeWindow("Rectified Stereo Pair", 1600,900);
    cv::imshow("Rectified Stereo Pair",visualization);
    cv::waitKey(0);
}



RectificationResult StereoRectifier::Rectify(const StereoImagePair& imagePair, const CameraIntrinsics& intrinsics, const StereoGeometry& geometry) const
{
    std::cout << "Stereo rectification started." << std::endl;

    if (imagePair.leftImage.empty() || imagePair.rightImage.empty())
    {
        throw std::runtime_error("Cannot rectify empty images.");
    }

    if (imagePair.leftImage.size() != imagePair.rightImage.size())
    {
        throw std::runtime_error("Left and right images must have equal dimensions.");
    }

    const cv::Size imageSize = imagePair.leftImage.size();
    const cv::Mat cameraMatrix = CreateOpenCvCameraMatrix(intrinsics);
    const cv::Mat distortionCoefficients = CreateZeroDistortionCoefficients();
    const cv::Mat rotation =ConvertToOpenCvMatrix(geometry.rotation);
    const cv::Mat translation = ConvertToOpenCvVector(geometry.translation);

    cv::Mat leftRectificationRotation;
    cv::Mat rightRectificationRotation;

    cv::Mat leftProjectionMatrix;
    cv::Mat rightProjectionMatrix;

    cv::Mat reprojectionMatrixQ;

    cv::stereoRectify(
        cameraMatrix,
        distortionCoefficients,
        cameraMatrix,
        distortionCoefficients,
        imageSize,
        rotation,
        translation,
        leftRectificationRotation,
        rightRectificationRotation,
        leftProjectionMatrix,
        rightProjectionMatrix,
        reprojectionMatrixQ,
        cv::CALIB_ZERO_DISPARITY,
        0.0);

    cv::Mat leftMapX;
    cv::Mat leftMapY;

    cv::Mat rightMapX;
    cv::Mat rightMapY;

    cv::initUndistortRectifyMap(
        cameraMatrix,
        distortionCoefficients,
        leftRectificationRotation,
        leftProjectionMatrix,
        imageSize,
        CV_32FC1,
        leftMapX,
        leftMapY);

    cv::initUndistortRectifyMap(
        cameraMatrix,
        distortionCoefficients,
        rightRectificationRotation,
        rightProjectionMatrix,
        imageSize,
        CV_32FC1,
        rightMapX,
        rightMapY);

    RectificationResult result;

    cv::remap(
        imagePair.leftImage,
        result.rectifiedLeftImage,
        leftMapX,
        leftMapY,
        cv::INTER_LINEAR);

    cv::remap(
        imagePair.rightImage,
        result.rectifiedRightImage,
        rightMapX,
        rightMapY,
        cv::INTER_LINEAR);

    result.reprojectionMatrixQ = ConvertToEigenMatrix4d(reprojectionMatrixQ);
    result.leftRectificationRotation = ConvertToEigenMatrix3d(leftRectificationRotation);
    result.leftProjectionMatrix = ConvertToEigenMatrix3x4d(leftProjectionMatrix);

    std::cout << "Stereo rectification finished successfully." << std::endl;

    return result;
}

cv::Mat StereoRectifier::CreateOpenCvCameraMatrix(const CameraIntrinsics& intrinsics) const
{
    return (
        cv::Mat_<double>(3, 3) <<
        intrinsics.fx, 0.0, intrinsics.cx,
        0.0, intrinsics.fy, intrinsics.cy,
        0.0, 0.0, 1.0
        );
}

cv::Mat StereoRectifier::CreateZeroDistortionCoefficients() const
{
    return cv::Mat::zeros(1, 5, CV_64F);
}

Eigen::Matrix4d StereoRectifier::ConvertToEigenMatrix4d(const cv::Mat& matrix) const
{
    if (matrix.rows != 4 || matrix.cols != 4)
    {
        throw std::runtime_error("Expected a 4x4 matrix.");
    }

    cv::Mat matrixAsDouble;
    matrix.convertTo(matrixAsDouble, CV_64F);

    Eigen::Matrix4d result;

    for (int row = 0; row < 4; ++row)
    {
        for (int column = 0; column < 4; ++column)
        {
            result(row, column) = matrixAsDouble.at<double>(row, column);
        }
    }

    return result;
}

cv::Mat StereoRectifier::ConvertToOpenCvMatrix(const Eigen::Matrix3d& matrix) const
{
    cv::Mat result(3, 3, CV_64F);

    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            result.at<double>(row, column) = matrix(row, column);
        }
    }

    return result;
}

cv::Mat StereoRectifier::ConvertToOpenCvVector(const Eigen::Vector3d& vector) const
{
    cv::Mat result(3, 1, CV_64F);

    for (int row = 0; row < 3; ++row)
    {
        result.at<double>(row, 0) = vector(row);
    }

    return result;
}

Eigen::Matrix3d StereoRectifier::ConvertToEigenMatrix3d(const cv::Mat& mat) const
{
    Eigen::Matrix3d eigenMat;

    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            eigenMat(r, c) = mat.at<double>(r, c);
        }
    }

    return eigenMat;
}

Eigen::Matrix<double, 3, 4> StereoRectifier::ConvertToEigenMatrix3x4d(const cv::Mat& mat) const
{
    Eigen::Matrix<double, 3, 4> eigenMat;

    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 4; ++c)
        {
            eigenMat(r, c) = mat.at<double>(r, c);
        }
    }

    return eigenMat;
}
