#include "DepthMapEvaluator.h"

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace
{
    Eigen::Vector2d ProjectThinPrismFisheyeNormalized(
        const Eigen::Vector2d& xy,
        const RawCameraCalibration& calib)
    {
        const double x = xy.x();
        const double y = xy.y();

        const double r = std::sqrt(x * x + y * y);
        const double theta = std::atan(r);

        double scale = 1.0;
        if (r > 1e-12)
        {
            scale = theta / r;
        }

        const double xf = scale * x;
        const double yf = scale * y;

        const double theta2 = theta * theta;
        const double theta4 = theta2 * theta2;
        const double theta6 = theta4 * theta2;
        const double theta8 = theta4 * theta4;

        const double radial =
            1.0 +
            calib.k1 * theta2 +
            calib.k2 * theta4 +
            calib.k3 * theta6 +
            calib.k4 * theta8;

        const double xDistorted =
            xf * radial +
            2.0 * calib.p1 * xf * yf +
            calib.p2 * (theta2 + 2.0 * xf * xf) +
            calib.sx1 * theta2;

        const double yDistorted =
            yf * radial +
            2.0 * calib.p2 * xf * yf +
            calib.p1 * (theta2 + 2.0 * yf * yf) +
            calib.sy1 * theta2;

        return Eigen::Vector2d(xDistorted, yDistorted);
    }

    Eigen::Vector2d UndistortRawPixelToNormalizedRay(
        double pixelX,
        double pixelY,
        const RawCameraCalibration& calib)
    {
        const Eigen::Vector2d target(
            (pixelX - calib.cx) / calib.fx,
            (pixelY - calib.cy) / calib.fy
        );

        Eigen::Vector2d xy = target;

        for (int iter = 0; iter < 10; ++iter)
        {
            const Eigen::Vector2d projected =
                ProjectThinPrismFisheyeNormalized(xy, calib);

            const Eigen::Vector2d residual = projected - target;

            if (residual.norm() < 1e-10)
            {
                break;
            }

            const double eps = 1e-6;
            Eigen::Matrix2d J;

            for (int i = 0; i < 2; ++i)
            {
                Eigen::Vector2d xyPlus = xy;
                Eigen::Vector2d xyMinus = xy;

                xyPlus(i) += eps;
                xyMinus(i) -= eps;

                const Eigen::Vector2d fPlus =
                    ProjectThinPrismFisheyeNormalized(xyPlus, calib);

                const Eigen::Vector2d fMinus =
                    ProjectThinPrismFisheyeNormalized(xyMinus, calib);

                J.col(i) = (fPlus - fMinus) / (2.0 * eps);
            }

            const Eigen::Vector2d delta =
                J.colPivHouseholderQr().solve(residual);

            xy -= delta;

            if (delta.norm() < 1e-10)
            {
                break;
            }
        }

        return xy;
    }

    Eigen::Vector3d BackProjectRawGtPixel(
        int pixelX,
        int pixelY,
        float depth,
        const RawCameraCalibration& calib)
    {
        const Eigen::Vector2d xy =
            UndistortRawPixelToNormalizedRay(
                static_cast<double>(pixelX),
                static_cast<double>(pixelY),
                calib
            );

        return Eigen::Vector3d(
            xy.x() * static_cast<double>(depth),
            xy.y() * static_cast<double>(depth),
            static_cast<double>(depth)
        );
    }
}

cv::Mat DepthMapEvaluator::RectifyGroundTruthDepth(const cv::Mat& gtDepthRaw, const RawCameraCalibration& rawCalibration, const RectificationResult& rectResult, const cv::Size& rectifiedSize)
{
    if (gtDepthRaw.empty())
    {
        throw std::runtime_error("Ground truth depth map is empty.");
    }

    if (gtDepthRaw.type() != CV_32FC1)
    {
        throw std::runtime_error("Ground truth depth map must be CV_32FC1.");
    }

    if (gtDepthRaw.cols != rawCalibration.imageWidth ||
        gtDepthRaw.rows != rawCalibration.imageHeight)
    {
        throw std::runtime_error("Ground truth depth size does not match raw calibration size.");
    }

    const float nanValue = std::numeric_limits<float>::quiet_NaN();

    cv::Mat gtDepthRectified(
        rectifiedSize,
        CV_32FC1,
        cv::Scalar(nanValue)
    );

    const Eigen::Matrix3d& R1 = rectResult.leftRectificationRotation;
    const Eigen::Matrix<double, 3, 4>& P1 = rectResult.leftProjectionMatrix;

    int validRawCount = 0;
    int projectedCount = 0;

    for (int y = 0; y < gtDepthRaw.rows; ++y)
    {
        const float* gtRow = gtDepthRaw.ptr<float>(y);

        for (int x = 0; x < gtDepthRaw.cols; ++x)
        {
            const float depth = gtRow[x];

            if (!std::isfinite(depth) || depth <= 0.0f)
            {
                continue;
            }

            ++validRawCount;

            const Eigen::Vector3d Xraw =
                BackProjectRawGtPixel(x, y, depth, rawCalibration);

            const Eigen::Vector3d Xrect = R1 * Xraw;

            if (Xrect.z() <= 0.0)
            {
                continue;
            }

            Eigen::Vector4d Xhomogeneous;
            Xhomogeneous << Xrect.x(), Xrect.y(), Xrect.z(), 1.0;

            const Eigen::Vector3d projected = P1 * Xhomogeneous;

            if (std::abs(projected.z()) < 1e-12)
            {
                continue;
            }

            const double u = projected.x() / projected.z();
            const double v = projected.y() / projected.z();

            const int uRounded = static_cast<int>(std::round(u));
            const int vRounded = static_cast<int>(std::round(v));

            if (uRounded < 0 || uRounded >= gtDepthRectified.cols ||
                vRounded < 0 || vRounded >= gtDepthRectified.rows)
            {
                continue;
            }

            float& targetDepth =
                gtDepthRectified.at<float>(vRounded, uRounded);

            const float rectifiedDepth =
                static_cast<float>(Xrect.z());

            // Z-buffer: if multiple GT points land on the same rectified pixel,
            // keep the closest one.
            if (!std::isfinite(targetDepth) || rectifiedDepth < targetDepth)
            {
                targetDepth = rectifiedDepth;
            }

            ++projectedCount;
        }
    }

    std::cout << "GT rectification finished." << std::endl;
    std::cout << "Valid raw GT points: " << validRawCount << std::endl;
    std::cout << "Projected rectified GT points: " << projectedCount << std::endl;

    return gtDepthRectified;
}
