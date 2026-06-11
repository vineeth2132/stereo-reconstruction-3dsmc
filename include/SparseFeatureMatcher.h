#pragma once
#include "DataLoader.h"

struct FeatureSet
{
    // keypoints[i] matches with descriptors.row(i)
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
};

struct StereoFeatureSet
{
    FeatureSet leftFeatures;
    FeatureSet rightFeatures;
};

struct SparseMatchingResult
{
    std::vector<cv::DMatch> matches;

    std::vector<cv::Point2f> leftMatchedPoints;
    std::vector<cv::Point2f> rightMatchedPoints;
};

class SparseFeatureMatcher
{
public:
    SparseFeatureMatcher();

    StereoFeatureSet DetectFeatures(const StereoImagePair& imgPair) const;
    void ShowKeypoints(const StereoImagePair& imgPair, const StereoFeatureSet& featureSet) const;

    // Decreasing the ratioThreshold decreases the number of matches, increases the confidence of matches
    SparseMatchingResult MatchFeatures(const StereoFeatureSet& featureSet, float ratioThreshold = 0.75f) const;
    void ShowMatches(const StereoImagePair& imgPair, const StereoFeatureSet& featureSet, const SparseMatchingResult& matchingResult) const;

private:
    cv::Ptr<cv::SIFT> siftDetector;
};