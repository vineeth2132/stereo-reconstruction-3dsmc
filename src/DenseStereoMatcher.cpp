#include "DenseStereoMatcher.h"

cv::Mat DenseStereoMatcher::ConvertToGrayscale(const cv::Mat& image) const
{
    if (image.empty()) { throw std::runtime_error("Cannot convert an empty image to grayscale."); }
    if (image.channels() == 1) { return image; }

    cv::Mat grayscaleImage;
    cv::cvtColor(image, grayscaleImage, cv::COLOR_BGR2GRAY);

    return grayscaleImage;
}

DenseMatchingResult DenseStereoMatcher::BuildResult(const cv::Mat& rawDisparity, const int minDisparity) const
{
    if (rawDisparity.empty()) { throw std::runtime_error("Cannot build result from an empty disparity map."); }
    DenseMatchingResult result;

    /*
        OpenCV stores BM and SGBM disparities as fixed-point values
        scaled by 16. Convert them to actual pixel disparities.
    */
    rawDisparity.convertTo(result.rawDisparity, CV_32F, 1.0 / 16.0);

    result.validDisparityMask = result.rawDisparity > static_cast<float>(minDisparity);
    cv::normalize(result.rawDisparity, result.disparityVisualization, 0, 255, cv::NORM_MINMAX, CV_8U, result.validDisparityMask);

    return result;
}

DenseMatchingResult DenseStereoMatcher::ComputeDisparity(const RectificationResult& rectificationResult, const DenseStereoConfig& config) const
{
    std::cout << "Disparity map computation started." << std::endl;

    if (config.numDisparities <= 0 || config.numDisparities % 16 != 0)
    {
        throw std::runtime_error("numDisparities must be a positive multiple of 16.");
    }

    if (config.blockSize <= 0 || config.blockSize % 2 == 0)
    {
        throw std::runtime_error("blockSize must be a positive odd number.");
    }

    const cv::Mat leftImage = ConvertToGrayscale(rectificationResult.rectifiedLeftImage);

    const cv::Mat rightImage = ConvertToGrayscale(rectificationResult.rectifiedRightImage);

    cv::Mat rawDisparity;

    if (config.method == DenseStereoMethod::StereoBM)
    {
        cv::Ptr<cv::StereoBM> matcher = cv::StereoBM::create(config.numDisparities, config.blockSize);
        matcher->setMinDisparity(config.minDisparity);
        matcher->compute(rightImage,leftImage, rawDisparity);
    }
    else
    {
        const int p1 = config.p1 > 0 ? config.p1 : 8 * config.blockSize * config.blockSize;
        const int p2 = config.p2 > 0 ? config.p2 : 32 * config.blockSize * config.blockSize;

        cv::Ptr<cv::StereoSGBM> matcher = cv::StereoSGBM::create(config.minDisparity, config.numDisparities, config.blockSize, p1, p2);
        matcher->setUniquenessRatio(10);
        matcher->setSpeckleWindowSize(100);
        matcher->setSpeckleRange(2);
        matcher->setDisp12MaxDiff(1);
        matcher->setPreFilterCap(31);
        matcher->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);
        matcher->compute(leftImage, rightImage, rawDisparity);
    }

    std::cout << "Disparity map computation finished successfully." << std::endl;

    return BuildResult(rawDisparity, config.minDisparity);
}

void DenseMatchingResult::ShowDisparity() const
{
    if (disparityVisualization.empty())
    {
        throw std::runtime_error("Cannot visualize an empty disparity map.");
    }

    cv::Mat coloredDisparity;
    cv::applyColorMap(disparityVisualization, coloredDisparity, cv::COLORMAP_JET);
    coloredDisparity.setTo(cv::Scalar(0, 0, 0), ~validDisparityMask);

    cv::namedWindow("Disparity Map", cv::WINDOW_NORMAL);
    cv::resizeWindow("Disparity Map", 1200, 800);
    cv::imshow("Disparity Map", coloredDisparity);

    cv::waitKey(0);
    cv::destroyAllWindows();
}