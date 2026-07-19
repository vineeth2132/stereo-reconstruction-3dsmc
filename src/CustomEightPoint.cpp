#include "CustomEightPoint.h"

#include <stdexcept>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <limits>
#include <iostream>

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

    /*
        Only vt (the 9 right singular vectors) is needed for the null-space
        solution; FULL_UV would additionally build the N x N left singular
        matrix, which is O(N^2) memory / O(N^3) time when N is the full inlier
        set (~23k matches on facade froze the pipeline for ~an hour here).
        The thin SVD only returns min(N, 9) rows in vt, so the minimal 8-row
        system still needs FULL_UV to expose the 9th (null-space) vector —
        there it is an 8x8 u, which costs nothing.
    */
    cv::Mat w, u, vt;
    const int svdFlags =
        cv::SVD::MODIFY_A | (A.rows < 9 ? cv::SVD::FULL_UV : 0);
    cv::SVD::compute(A, w, u, vt, svdFlags);

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
double CustomEightPoint::ComputeSampsonError(
    const cv::Mat& F,
    const cv::Point2f& point1,
    const cv::Point2f& point2)
{
    cv::Mat F64;
    F.convertTo(F64, CV_64F);

    const cv::Mat x1 = (cv::Mat_<double>(3, 1) <<
        point1.x,
        point1.y,
        1.0);

    const cv::Mat x2 = (cv::Mat_<double>(3, 1) <<
        point2.x,
        point2.y,
        1.0);

    const cv::Mat Fx1 = F64 * x1;
    const cv::Mat Ftx2 = F64.t() * x2;
    const cv::Mat x2tFx1 = x2.t() * F64 * x1;

    const double numerator =
        x2tFx1.at<double>(0, 0) * x2tFx1.at<double>(0, 0);

    const double denominator =
        Fx1.at<double>(0, 0) * Fx1.at<double>(0, 0) +
        Fx1.at<double>(1, 0) * Fx1.at<double>(1, 0) +
        Ftx2.at<double>(0, 0) * Ftx2.at<double>(0, 0) +
        Ftx2.at<double>(1, 0) * Ftx2.at<double>(1, 0);

    if (denominator < 1e-12)
        return std::numeric_limits<double>::max();

    return numerator / denominator;
}
CustomRansacResult CustomEightPoint::EstimateFundamentalRansac(
    const std::vector<cv::Point2f>& points1,
    const std::vector<cv::Point2f>& points2,
    int iterations,
    double sampsonThreshold)
{
    if (points1.size() != points2.size())
        throw std::runtime_error("Point vectors must have the same size.");

    if (points1.size() < 8)
        throw std::runtime_error("At least 8 point correspondences are required for RANSAC.");

    const int pointCount = static_cast<int>(points1.size());

    std::vector<int> indices(pointCount);
    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 rng(42);

    int bestInlierCount = 0;
    double bestMeanError = std::numeric_limits<double>::max();

    cv::Mat bestF;
    cv::Mat bestMask = cv::Mat::zeros(pointCount, 1, CV_8U);

    for (int iter = 0; iter < iterations; ++iter)
    {
        std::shuffle(indices.begin(), indices.end(), rng);

        std::vector<cv::Point2f> samplePoints1;
        std::vector<cv::Point2f> samplePoints2;

        samplePoints1.reserve(8);
        samplePoints2.reserve(8);

        for (int i = 0; i < 8; ++i)
        {
            const int idx = indices[i];
            samplePoints1.push_back(points1[idx]);
            samplePoints2.push_back(points2[idx]);
        }

        cv::Mat candidateF;

        try
        {
            candidateF = EstimateFundamental(samplePoints1, samplePoints2);
        }
        catch (...)
        {
            continue;
        }

        int currentInlierCount = 0;
        double currentErrorSum = 0.0;
        cv::Mat currentMask = cv::Mat::zeros(pointCount, 1, CV_8U);

        for (int i = 0; i < pointCount; ++i)
        {
            const double error =
                ComputeSampsonError(candidateF, points1[i], points2[i]);

            if (error < sampsonThreshold)
            {
                currentMask.at<uchar>(i, 0) = 1;
                currentInlierCount++;
                currentErrorSum += error;
            }
        }

        if (currentInlierCount < 8)
            continue;

        const double currentMeanError =
            currentErrorSum / static_cast<double>(currentInlierCount);

        if (currentInlierCount > bestInlierCount ||
            (currentInlierCount == bestInlierCount && currentMeanError < bestMeanError))
        {
            bestInlierCount = currentInlierCount;
            bestMeanError = currentMeanError;
            bestF = candidateF.clone();
            bestMask = currentMask.clone();
        }
    }

    if (bestInlierCount < 8 || bestF.empty())
        throw std::runtime_error("Custom RANSAC failed to find a valid fundamental matrix.");

    std::vector<cv::Point2f> finalInlierPoints1;
    std::vector<cv::Point2f> finalInlierPoints2;

    finalInlierPoints1.reserve(bestInlierCount);
    finalInlierPoints2.reserve(bestInlierCount);

    for (int i = 0; i < pointCount; ++i)
    {
        if (bestMask.at<uchar>(i, 0) != 0)
        {
            finalInlierPoints1.push_back(points1[i]);
            finalInlierPoints2.push_back(points2[i]);
        }
    }

    const cv::Mat refinedF =
        EstimateFundamental(finalInlierPoints1, finalInlierPoints2);

    double refinedErrorSum = 0.0;

    for (size_t i = 0; i < finalInlierPoints1.size(); ++i)
    {
        refinedErrorSum += ComputeSampsonError(
            refinedF,
            finalInlierPoints1[i],
            finalInlierPoints2[i]
        );
    }

    CustomRansacResult result;
    result.fundamentalMatrix = refinedF;
    result.inlierMask = bestMask;
    result.inlierCount = bestInlierCount;
    result.meanSampsonError =
        refinedErrorSum / static_cast<double>(bestInlierCount);

    return result;
}