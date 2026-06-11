#include "SparseFeatureMatcher.h"

SparseFeatureMatcher::SparseFeatureMatcher() 
{
	siftDetector = cv::SIFT::create();
};

StereoFeatureSet SparseFeatureMatcher::DetectFeatures(const StereoImagePair& imagePair) const
{
    std::cout << "SIFT feature detection started." << std::endl;

    if (imagePair.leftImage.empty() || imagePair.rightImage.empty())
    {
        throw std::runtime_error(
            "Cannot detect features: one or both images are empty.");
    }

    StereoFeatureSet featureSet;

    siftDetector->detectAndCompute(
        imagePair.leftImage,
        cv::noArray(),
        featureSet.leftFeatures.keypoints,
        featureSet.leftFeatures.descriptors);

    siftDetector->detectAndCompute(
        imagePair.rightImage,
        cv::noArray(),
        featureSet.rightFeatures.keypoints,
        featureSet.rightFeatures.descriptors);

    std::cout << "SIFT feature detection finished. Number of keypoints: Left = " << featureSet.leftFeatures.keypoints.size() << " - Right = " << featureSet.rightFeatures.keypoints.size() << std::endl;

    return featureSet;
}

void SparseFeatureMatcher::ShowKeypoints( const StereoImagePair& imagePair, const StereoFeatureSet& featureSet) const
{
    cv::Mat leftVisualization;
    cv::Mat rightVisualization;

    cv::drawKeypoints(
        imagePair.leftImage,
        featureSet.leftFeatures.keypoints,
        leftVisualization,
        cv::Scalar::all(-1),
        cv::DrawMatchesFlags::DEFAULT);

    cv::drawKeypoints(
        imagePair.rightImage,
        featureSet.rightFeatures.keypoints,
        rightVisualization,
        cv::Scalar::all(-1),
        cv::DrawMatchesFlags::DEFAULT);

    cv::namedWindow("Left SIFT Keypoints", cv::WINDOW_NORMAL);
    cv::namedWindow("Right SIFT Keypoints", cv::WINDOW_NORMAL);

    cv::resizeWindow("Left SIFT Keypoints", 1000, 700);
    cv::resizeWindow("Right SIFT Keypoints", 1000, 700);

    cv::imshow("Left SIFT Keypoints", leftVisualization);
    cv::imshow("Right SIFT Keypoints", rightVisualization);

    cv::waitKey(0);
}

SparseMatchingResult SparseFeatureMatcher::MatchFeatures( const StereoFeatureSet& featureSet, const float ratioThreshold) const
{
    std::cout << "SIFT correspondence matching started." << std::endl;

    if (featureSet.leftFeatures.descriptors.empty() || featureSet.rightFeatures.descriptors.empty())
    {
        throw std::runtime_error("Cannot match features: descriptors are empty.");
    }

    cv::BFMatcher matcher(cv::NORM_L2);

    std::vector<std::vector<cv::DMatch>> nearestMatches;

    matcher.knnMatch(featureSet.leftFeatures.descriptors, featureSet.rightFeatures.descriptors, nearestMatches, 2);

    SparseMatchingResult result;

    for (const std::vector<cv::DMatch>& candidateMatches : nearestMatches)
    {
        if (candidateMatches.size() < 2) { continue; }

        const cv::DMatch& bestMatch = candidateMatches[0];
        const cv::DMatch& secondBestMatch = candidateMatches[1];

        if (bestMatch.distance < ratioThreshold * secondBestMatch.distance)
        {
            result.matches.push_back(bestMatch);

            result.leftMatchedPoints.push_back(featureSet.leftFeatures.keypoints[bestMatch.queryIdx].pt);
            result.rightMatchedPoints.push_back(featureSet.rightFeatures.keypoints[bestMatch.trainIdx].pt);
        }
    }

    std::cout << "SIFT correspondence matching finished. Number of Matches: " << result.matches.size() << std::endl;

    return result;
}

void SparseFeatureMatcher::ShowMatches(const StereoImagePair& imagePair, const StereoFeatureSet& featureSet, const SparseMatchingResult& matchingResult) const
{
    cv::Mat visualization;

    cv::drawMatches(
        imagePair.leftImage,
        featureSet.leftFeatures.keypoints,
        imagePair.rightImage,
        featureSet.rightFeatures.keypoints,
        matchingResult.matches,
        visualization,
        cv::Scalar(0, 255, 0),
        cv::Scalar(0, 0, 255),
        std::vector<char>(),
        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    cv::namedWindow("SIFT Matches", cv::WINDOW_NORMAL);
    cv::resizeWindow("SIFT Matches", 1600, 900);

    cv::imshow("SIFT Matches", visualization);
    cv::waitKey(0);
}
