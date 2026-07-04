#include "StereoGeometryEstimator.h"
#include "CustomEightPoint.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace
{
    double ComputeMeanSampsonError(
        const cv::Mat& F,
        const std::vector<cv::Point2f>& points1,
        const std::vector<cv::Point2f>& points2)
    {
        double totalError = 0.0;
        int validCount = 0;

        cv::Mat F64;
        F.convertTo(F64, CV_64F);

        for (size_t i = 0; i < points1.size(); ++i)
        {
            cv::Mat x1 = (cv::Mat_<double>(3, 1) <<
                points1[i].x,
                points1[i].y,
                1.0);

            cv::Mat x2 = (cv::Mat_<double>(3, 1) <<
                points2[i].x,
                points2[i].y,
                1.0);

            cv::Mat Fx1 = F64 * x1;
            cv::Mat Ftx2 = F64.t() * x2;
            cv::Mat x2tFx1 = x2.t() * F64 * x1;

            const double numerator =
                x2tFx1.at<double>(0, 0) * x2tFx1.at<double>(0, 0);

            const double denominator =
                Fx1.at<double>(0, 0) * Fx1.at<double>(0, 0) +
                Fx1.at<double>(1, 0) * Fx1.at<double>(1, 0) +
                Ftx2.at<double>(0, 0) * Ftx2.at<double>(0, 0) +
                Ftx2.at<double>(1, 0) * Ftx2.at<double>(1, 0);

            if (denominator < 1e-12)
            {
                continue;
            }

            totalError += numerator / denominator;
            validCount++;
        }

        if (validCount == 0)
        {
            return -1.0;
        }

        return totalError / validCount;
    }

    std::vector<cv::Point2f> SelectInlierPoints(
        const std::vector<cv::Point2f>& points,
        const cv::Mat& inlierMask)
    {
        std::vector<cv::Point2f> inlierPoints;

        for (int i = 0; i < inlierMask.rows; ++i)
        {
            if (inlierMask.at<unsigned char>(i, 0) != 0)
            {
                inlierPoints.push_back(points[static_cast<size_t>(i)]);
            }
        }

        return inlierPoints;
    }

    void PrintSingularValuesOfF(const cv::Mat& F, const std::string& name)
    {
        cv::Mat F64;
        F.convertTo(F64, CV_64F);

        cv::Mat w, u, vt;
        cv::SVD::compute(F64, w, u, vt);

        std::cout << name << " singular values:\n" << w << "\n";
    }
}

void StereoGeometry::PrintData() const
{
    int inlierCount = 0;

    for (const unsigned char isInlier : inlierMask)
    {
        if (isInlier)
        {
            ++inlierCount;
        }
    }

    const double inlierRatio =
        inlierMask.empty()
            ? 0.0
            : static_cast<double>(inlierCount) / static_cast<double>(inlierMask.size());

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

StereoGeometry StereoGeometryEstimator::EstimateGeometry(
    const SparseMatchingResult& matchingResult,
    const CameraIntrinsics& intrinsics) const
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

    /*
        OpenCV fundamental matrix is still computed only for comparison.
        The actual pipeline below uses the custom RANSAC + custom 8-point result.
    */
    cv::Mat openCvInlierMask;

    const cv::Mat openCvFundamentalMatrix = cv::findFundamentalMat(
        leftPoints,
        rightPoints,
        cv::FM_RANSAC,
        1.0,
        0.999,
        openCvInlierMask
    );

    if (openCvFundamentalMatrix.empty())
    {
        throw std::runtime_error("OpenCV fundamental matrix estimation failed.");
    }

    /*
        Test 1:
        Run custom normalized 8-point using OpenCV's inliers only.
        This checks whether the 8-point algorithm itself works on clean matches.
    */
    const std::vector<cv::Point2f> leftInlierPoints =
        SelectInlierPoints(leftPoints, openCvInlierMask);

    const std::vector<cv::Point2f> rightInlierPoints =
        SelectInlierPoints(rightPoints, openCvInlierMask);

    if (leftInlierPoints.size() >= 8 && rightInlierPoints.size() >= 8)
    {
        const cv::Mat customFundamentalMatrix =
            CustomEightPoint::EstimateFundamental(leftInlierPoints, rightInlierPoints);

        const double opencvSampsonError =
            ComputeMeanSampsonError(
                openCvFundamentalMatrix,
                leftInlierPoints,
                rightInlierPoints
            );

        const double customSampsonError =
            ComputeMeanSampsonError(
                customFundamentalMatrix,
                leftInlierPoints,
                rightInlierPoints
            );

        std::cout << "\n===== Custom 8-Point Check =====\n";
        std::cout << "OpenCV Fundamental Matrix:\n"
                  << openCvFundamentalMatrix << "\n\n";

        std::cout << "Custom Fundamental Matrix:\n"
                  << customFundamentalMatrix << "\n\n";

        std::cout << "OpenCV mean Sampson error on OpenCV inliers: "
                  << opencvSampsonError << "\n";

        std::cout << "Custom mean Sampson error on OpenCV inliers: "
                  << customSampsonError << "\n";

        PrintSingularValuesOfF(openCvFundamentalMatrix, "OpenCV F");
        PrintSingularValuesOfF(customFundamentalMatrix, "Custom F");

        std::cout << "================================\n\n";
    }
    else
    {
        std::cout << "Not enough inliers to test custom 8-point algorithm.\n";
    }

    /*
        Test 2:
        Run custom RANSAC + custom normalized 8-point on all raw matches.
        This is the custom replacement for OpenCV findFundamentalMat(..., RANSAC).
    */
    const CustomRansacResult customRansacResult =
        CustomEightPoint::EstimateFundamentalRansac(
            leftPoints,
            rightPoints,
            2000,
            1.0
        );

    std::cout << "\n===== Custom RANSAC + 8-Point Check =====\n";

    std::cout << "OpenCV inliers: "
              << cv::countNonZero(openCvInlierMask)
              << " / "
              << leftPoints.size()
              << "\n";

    std::cout << "Custom RANSAC inliers: "
              << customRansacResult.inlierCount
              << " / "
              << leftPoints.size()
              << "\n";

    std::cout << "Custom RANSAC mean Sampson error: "
              << customRansacResult.meanSampsonError
              << "\n";

    std::cout << "Custom RANSAC Fundamental Matrix:\n"
              << customRansacResult.fundamentalMatrix
              << "\n";

    PrintSingularValuesOfF(
        customRansacResult.fundamentalMatrix,
        "Custom RANSAC F"
    );

    std::cout << "========================================\n\n";

    /*
        IMPORTANT:
        From here onward, the pipeline uses the custom fundamental matrix
        and the custom inlier mask.
    */
    const cv::Mat fundamentalMatrix = customRansacResult.fundamentalMatrix;
    cv::Mat inlierMask = customRansacResult.inlierMask.clone();

    std::cout << "Using CUSTOM RANSAC + CUSTOM 8-point fundamental matrix in pipeline."
              << std::endl;

    const cv::Mat cameraMatrix = CreateOpenCvCameraMatrix(intrinsics);

    const cv::Mat essentialMatrix =
        cameraMatrix.t() * fundamentalMatrix * cameraMatrix;

    cv::Mat rotation;
    cv::Mat translation;

    const int recoveredPointCount = cv::recoverPose(
        essentialMatrix,
        leftPoints,
        rightPoints,
        cameraMatrix,
        rotation,
        translation,
        inlierMask
    );

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

cv::Mat StereoGeometryEstimator::CreateOpenCvCameraMatrix(
    const CameraIntrinsics& intrinsics) const
{
    return (
        cv::Mat_<double>(3, 3) <<
        intrinsics.fx, 0.0, intrinsics.cx,
        0.0, intrinsics.fy, intrinsics.cy,
        0.0, 0.0, 1.0
    );
}

Eigen::Matrix3d StereoGeometryEstimator::ConvertToEigenMatrix3d(
    const cv::Mat& matrix) const
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

Eigen::Vector3d StereoGeometryEstimator::ConvertToEigenVector3d(
    const cv::Mat& vector) const
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