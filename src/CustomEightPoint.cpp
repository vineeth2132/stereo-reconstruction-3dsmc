#include "CustomEightPoint.h"

#include <stdexcept>
#include <cmath>

void CustomEightPoint::NormalizePoints(
    const std::vector<cv::Point2f>& points,
    std::vector<cv::Point2f>& normalizedPoints,
    cv::Mat& T)
{
    if (points.empty())
        throw std::runtime_error("No points given for normalization.");

    double meanX = 0.0;
    double meanY = 0.0;

    for (const auto& p : points)
    {
        meanX += p.x;
        meanY += p.y;
    }

    meanX /= static_cast<double>(points.size());
    meanY /= static_cast<double>(points.size());

    double meanDistance = 0.0;

    for (const auto& p : points)
    {
        const double dx = p.x - meanX;
        const double dy = p.y - meanY;
        meanDistance += std::sqrt(dx * dx + dy * dy);
    }

    meanDistance /= static_cast<double>(points.size());

    if (meanDistance < 1e-8)
        throw std::runtime_error("Mean distance too small during normalization.");

    const double scale = std::sqrt(2.0) / meanDistance;

    T = (cv::Mat_<double>(3, 3) <<
         scale, 0.0, -scale * meanX,
         0.0, scale, -scale * meanY,
         0.0, 0.0, 1.0);

    normalizedPoints.clear();
    normalizedPoints.reserve(points.size());

    for (const auto& p : points)
    {
        const double x = scale * (p.x - meanX);
        const double y = scale * (p.y - meanY);
        normalizedPoints.emplace_back(static_cast<float>(x), static_cast<float>(y));
    }
}

cv::Mat CustomEightPoint::EstimateFundamental(
    const std::vector<cv::Point2f>& points1,
    const std::vector<cv::Point2f>& points2)
{
    if (points1.size() != points2.size())
        throw std::runtime_error("Point vectors must have the same size.");

    if (points1.size() < 8)
        throw std::runtime_error("At least 8 point correspondences are required.");

    std::vector<cv::Point2f> normPoints1;
    std::vector<cv::Point2f> normPoints2;

    cv::Mat T1;
    cv::Mat T2;

    NormalizePoints(points1, normPoints1, T1);
    NormalizePoints(points2, normPoints2, T2);

    cv::Mat A(static_cast<int>(points1.size()), 9, CV_64F);

    for (int i = 0; i < static_cast<int>(points1.size()); ++i)
    {
        const double x1 = normPoints1[i].x;
        const double y1 = normPoints1[i].y;
        const double x2 = normPoints2[i].x;
        const double y2 = normPoints2[i].y;

        A.at<double>(i, 0) = x2 * x1;
        A.at<double>(i, 1) = x2 * y1;
        A.at<double>(i, 2) = x2;
        A.at<double>(i, 3) = y2 * x1;
        A.at<double>(i, 4) = y2 * y1;
        A.at<double>(i, 5) = y2;
        A.at<double>(i, 6) = x1;
        A.at<double>(i, 7) = y1;
        A.at<double>(i, 8) = 1.0;
    }

    cv::Mat w, u, vt;
    cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    cv::Mat f = vt.row(8).reshape(0, 3);

    cv::SVD::compute(f, w, u, vt);

    w.at<double>(2) = 0.0;

    cv::Mat F_rank2 = u * cv::Mat::diag(w) * vt;

    cv::Mat F = T2.t() * F_rank2 * T1;

    double normValue = cv::norm(F);

    if (normValue > 1e-8)
        F = F / normValue;

    return F;
}