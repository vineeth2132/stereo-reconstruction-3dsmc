#include "StereoGeometryEstimator.h"

void StereoGeometry::PrintData() const
{
    int inlierCount = 0;

    for (const unsigned char isInlier : inlierMask)
    {
        if (isInlier) { ++inlierCount; }
    }
    const double inlierRatio = inlierMask.empty() ? 0.0 : static_cast<double>(inlierCount) / static_cast<double>(inlierMask.size());
    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(essentialMatrix);

    std::cout << "Fundamental matrix:\n" << fundamentalMatrix << "\n\n";
    std::cout << "Essential matrix:\n" << essentialMatrix << "\n\n";
    std::cout << "Rotation matrix:\n" << rotation << "\n\n";
    std::cout << "Translation direction:\n" << translation << "\n\n";
    std::cout << "det(R): " << rotation.determinant() << '\n';
    std::cout << "||t||: " << translation.norm() << '\n';
    std::cout << "Singular values of E: " << svd.singularValues().transpose() << '\n';
    std::cout << "Inliers: " << inlierCount << " / " << inlierMask.size() << '\n';
    std::cout << "Inlier ratio: " << inlierRatio * 100.0 << "%\n";
}

StereoGeometry StereoGeometryEstimator::EstimateGeometry(const SparseMatchingResult& matchingResult, const CameraIntrinsics& intrinsics) const
{
    std::cout << "Stereo geometry estimation started." << std::endl;

    const std::vector<cv::Point2f>& leftPoints = matchingResult.leftMatchedPoints;

    const std::vector<cv::Point2f>& rightPoints = matchingResult.rightMatchedPoints;

    if (leftPoints.size() != rightPoints.size())
    {
        throw std::runtime_error("Left and right matched-point counts must be equal.");
    }

    if (leftPoints.size() < 8)
    {
        throw std::runtime_error("At least eight matched point pairs are required.");
    }

    cv::Mat inlierMask;
    const cv::Mat fundamentalMatrix = cv::findFundamentalMat(leftPoints, rightPoints, cv::FM_RANSAC, 1.0, 0.999, inlierMask);

    if (fundamentalMatrix.empty())
    {
        throw std::runtime_error("Fundamental matrix estimation failed.");
    }

    const cv::Mat cameraMatrix = CreateOpenCvCameraMatrix(intrinsics);

    const cv::Mat essentialMatrix = cameraMatrix.t() * fundamentalMatrix * cameraMatrix;

    cv::Mat rotation;
    cv::Mat translation;

    const int recoveredPointCount = cv::recoverPose(essentialMatrix, leftPoints, rightPoints, cameraMatrix, rotation, translation, inlierMask);

    if (recoveredPointCount == 0)
    {
        throw std::runtime_error("Relative pose recovery failed.");
    }

    std::vector<unsigned char> mask;

    mask.reserve(inlierMask.total());

    for (int index = 0; index < inlierMask.rows; ++index)
    {
        mask.push_back(inlierMask.at<unsigned char>(index, 0));
    }


    StereoGeometry stereoGeometry;
    stereoGeometry.fundamentalMatrix = ConvertToEigenMatrix3d(fundamentalMatrix);
    stereoGeometry.essentialMatrix = ConvertToEigenMatrix3d(essentialMatrix);
    stereoGeometry.rotation = ConvertToEigenMatrix3d(rotation);
    stereoGeometry.translation = ConvertToEigenVector3d(translation);
    stereoGeometry.inlierMask = std::move(mask);

    std::cout << "Stereo geometry estimation finished successfully." << std::endl;

    return stereoGeometry;
}


cv::Mat StereoGeometryEstimator::CreateOpenCvCameraMatrix(const CameraIntrinsics& intrinsics) const
{
    return (
        cv::Mat_<double>(3, 3) <<
        intrinsics.fx, 0.0, intrinsics.cx,
        0.0, intrinsics.fy, intrinsics.cy,
        0.0, 0.0, 1.0
        );
}

Eigen::Matrix3d StereoGeometryEstimator::ConvertToEigenMatrix3d(const cv::Mat& matrix) const
{
    if (matrix.rows != 3 || matrix.cols != 3)
    {
        throw std::runtime_error("Expected a 3x3 matrix.");
    }

    cv::Mat matrixAsDouble;
    matrix.convertTo(matrixAsDouble, CV_64F);

    Eigen::Matrix3d eigenMatrix;

    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            eigenMatrix(row, column) = matrixAsDouble.at<double>(row, column);
        }
    }

    return eigenMatrix;
}

Eigen::Vector3d StereoGeometryEstimator::ConvertToEigenVector3d(const cv::Mat& vector) const
{
    if (vector.rows != 3 || vector.cols != 1)
    {
        throw std::runtime_error("Expected a 3x1 vector.");
    }

    cv::Mat vectorAsDouble;
    vector.convertTo(vectorAsDouble, CV_64F);

    Eigen::Vector3d result;

    for (int row = 0; row < 3; ++row)
    {
        result(row) = vectorAsDouble.at<double>(row, 0);
    }

    return result;
}
