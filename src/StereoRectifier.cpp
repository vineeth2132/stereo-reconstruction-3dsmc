#include "StereoRectifier.h"

//#define USE_OPENCV_RECTIFICATION

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

    RectificationResult result;

    const cv::Size imageSize = imagePair.leftImage.size();
    const cv::Mat cameraMatrix = CreateOpenCvCameraMatrix(intrinsics);
    const cv::Mat distortionCoefficients = CreateZeroDistortionCoefficients();
    const cv::Mat rotation =ConvertToOpenCvMatrix(geometry.rotation);
    const cv::Mat translation = ConvertToOpenCvVector(geometry.translation);

    bool useOpenCV = false;
    #ifdef USE_OPENCV_RECTIFICATION
        useOpenCV = true;
    #endif // USE_OPENCV_RECTIFICATION

    if (useOpenCV)
    {
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

        //std::cout << "--- OpenCV Rectification Matrices ---" << std::endl;
        //std::cout << "R1 (Left Rotation):\n" << leftRectificationRotation << "\n\n";
        //std::cout << "R2 (Right Rotation):\n" << rightRectificationRotation << "\n\n";
        //std::cout << "P1 (Left Projection):\n" << leftProjectionMatrix << "\n\n";
        //std::cout << "P2 (Right Projection):\n" << rightProjectionMatrix << "\n\n";
        //std::cout << "Q (Reprojection Matrix):\n" << reprojectionMatrixQ << "\n\n";

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
    }
    else
    {
        Eigen::Matrix3d eigenR1 = Eigen::Matrix3d::Identity();
        Eigen::Matrix3d eigenR2 = Eigen::Matrix3d::Identity();
        Eigen::Matrix<double, 3, 4> eigenP1 = Eigen::Matrix<double, 3, 4>::Zero();
        Eigen::Matrix<double, 3, 4> eigenP2 = Eigen::Matrix<double, 3, 4>::Zero();
        Eigen::Matrix4d eigenQ = Eigen::Matrix4d::Identity();

        CustomStereoRectify(
            intrinsics,
            geometry.rotation,
            geometry.translation,
            eigenR1, eigenR2, eigenP1, eigenP2, eigenQ
        );

        //std::cout << "======= CUSTOM EIGEN MATRICES =======" << std::endl;
        //std::cout << "R1 (Left Rotation):\n" << eigenR1 << "\n\n";
        //std::cout << "R2 (Right Rotation):\n" << eigenR2 << "\n\n";
        //std::cout << "P1 (Left Projection):\n" << eigenP1 << "\n\n";
        //std::cout << "P2 (Right Projection):\n" << eigenP2 << "\n\n";
        //std::cout << "Q (Reprojection Matrix):\n" << eigenQ << "\n\n";

        cv::Mat leftMapX, leftMapY, rightMapX, rightMapY;
        CreateRectificationMap(intrinsics, eigenR1, eigenP1, imageSize, leftMapX, leftMapY);
        CreateRectificationMap(intrinsics, eigenR2, eigenP2, imageSize, rightMapX, rightMapY);

        BilinearRemap(imagePair.leftImage, result.rectifiedLeftImage, leftMapX, leftMapY);
        BilinearRemap(imagePair.rightImage, result.rectifiedRightImage, rightMapX, rightMapY);

        result.reprojectionMatrixQ = eigenQ;
        result.leftRectificationRotation = eigenR1;
        result.leftProjectionMatrix = eigenP1;
    }

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

void StereoRectifier::CustomStereoRectify(
    const CameraIntrinsics& intrinsics,
    const Eigen::Matrix3d& R,
    const Eigen::Vector3d& t,
    Eigen::Matrix3d& R1,
    Eigen::Matrix3d& R2,
    Eigen::Matrix<double, 3, 4>& P1,
    Eigen::Matrix<double, 3, 4>& P2,
    Eigen::Matrix4d& Q) const
{
    Eigen::AngleAxisd angleAxis(R);
    Eigen::Vector3d rotVector = angleAxis.axis() * angleAxis.angle();

    Eigen::Matrix3d R_r = Eigen::Matrix3d::Identity();
    if (rotVector.norm() > 1e-8) {
        R_r = Eigen::AngleAxisd(rotVector.norm() * -0.5, rotVector.normalized()).toRotationMatrix();
    }

    Eigen::Matrix3d R_l = R_r.transpose();

    Eigen::Vector3d t_half = R_r * t;
    Eigen::Vector3d e1 = t_half.normalized();

    if (e1.x() < 0) e1 = -e1;

    Eigen::Vector3d e2(-e1.y(), e1.x(), 0.0);
    if (e2.norm() < 1e-6) {
        e2 = Eigen::Vector3d(0.0, 1.0, 0.0);
    }
    else {
        e2.normalize();
    }

    Eigen::Vector3d e3 = e1.cross(e2).normalized();

    Eigen::Matrix3d R_rect;
    R_rect.row(0) = e1;
    R_rect.row(1) = e2;
    R_rect.row(2) = e3;

    R1 = R_rect * R_l;
    R2 = R_rect * R_r;

    double f = intrinsics.fx;
    double cx = intrinsics.cx;
    double cy = intrinsics.cy;
    double baseline = t.norm();

    P1.setZero();
    P1(0, 0) = f;   P1(1, 1) = f;
    P1(0, 2) = cx;  P1(1, 2) = cy;
    P1(2, 2) = 1.0;

    P2 = P1;
    P2(0, 3) = -f * baseline;

    Q.setIdentity();
    Q(0, 0) = 1.0;
    Q(1, 1) = 1.0;
    Q(0, 3) = -cx;
    Q(1, 3) = -cy;
    Q(2, 2) = 0.0;
    Q(2, 3) = f;
    Q(3, 3) = 0.0;
    Q(3, 2) = 1.0 / baseline;
}

void StereoRectifier::CreateRectificationMap(
    const CameraIntrinsics& intrinsics,
    const Eigen::Matrix3d& R_rect,
    const Eigen::Matrix<double, 3, 4>& P_rect,
    const cv::Size& imageSize,
    cv::Mat& mapX,
    cv::Mat& mapY) const
{
    mapX.create(imageSize, CV_32FC1);
    mapY.create(imageSize, CV_32FC1);

    Eigen::Matrix3d K_old;
    K_old << intrinsics.fx, 0.0, intrinsics.cx,
        0.0, intrinsics.fy, intrinsics.cy,
        0.0, 0.0, 1.0;

    Eigen::Matrix3d K_new = P_rect.block<3, 3>(0, 0);

    Eigen::Matrix3d H = K_old * R_rect.transpose() * K_new.inverse();

    for (int v = 0; v < imageSize.height; ++v) {
        for (int u = 0; u < imageSize.width; ++u) {

            Eigen::Vector3d p_new(u, v, 1.0);

            Eigen::Vector3d p_old = H * p_new;

            mapX.at<float>(v, u) = static_cast<float>(p_old.x() / p_old.z());
            mapY.at<float>(v, u) = static_cast<float>(p_old.y() / p_old.z());
        }
    }
}

void StereoRectifier::BilinearRemap(const cv::Mat& src, cv::Mat& dst, const cv::Mat& mapX, const cv::Mat& mapY) const
{
    dst.create(mapX.size(), src.type());
    dst.setTo(cv::Scalar::all(0));

    int channels = src.channels();

    for (int v = 0; v < dst.rows; ++v) {
        for (int u = 0; u < dst.cols; ++u) {
            float x = mapX.at<float>(v, u);
            float y = mapY.at<float>(v, u);

            int x0 = static_cast<int>(std::floor(x));
            int y0 = static_cast<int>(std::floor(y));
            int x1 = x0 + 1;
            int y1 = y0 + 1;

            if (x0 >= 0 && y0 >= 0 && x1 < src.cols && y1 < src.rows) {
                float dx = x - x0;
                float dy = y - y0;

                float w00 = (1.0f - dx) * (1.0f - dy);
                float w10 = dx * (1.0f - dy);
                float w01 = (1.0f - dx) * dy;
                float w11 = dx * dy;

                if (channels == 1) {
                    float val = w00 * src.at<uchar>(y0, x0) + w10 * src.at<uchar>(y0, x1) +
                        w01 * src.at<uchar>(y1, x0) + w11 * src.at<uchar>(y1, x1);
                    dst.at<uchar>(v, u) = cv::saturate_cast<uchar>(val);
                }
                else if (channels == 3) {
                    cv::Vec3b c00 = src.at<cv::Vec3b>(y0, x0);
                    cv::Vec3b c10 = src.at<cv::Vec3b>(y0, x1);
                    cv::Vec3b c01 = src.at<cv::Vec3b>(y1, x0);
                    cv::Vec3b c11 = src.at<cv::Vec3b>(y1, x1);

                    cv::Vec3b out_pixel;
                    for (int c = 0; c < 3; ++c) {
                        float val = w00 * c00[c] + w10 * c10[c] + w01 * c01[c] + w11 * c11[c];
                        out_pixel[c] = cv::saturate_cast<uchar>(val);
                    }
                    dst.at<cv::Vec3b>(v, u) = out_pixel;
                }
            }
        }
    }
}

