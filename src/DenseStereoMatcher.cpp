#include "DenseStereoMatcher.h"
#include "CustomDenseMatcher.h"

#ifdef HAVE_OPENCV_XIMGPROC
#include <opencv2/ximgproc.hpp>
#endif

std::unique_ptr<IDenseStereoMatcher> CreateDenseMatcher(const DenseStereoConfig& config)
{
    switch (config.backend)
    {
    case DenseStereoBackend::Custom:
        return std::make_unique<CustomDenseMatcher>();

    case DenseStereoBackend::OpenCv:
    default:
        return std::make_unique<DenseStereoMatcher>();
    }
}

cv::Mat DenseStereoMatcher::ConvertToGrayscale(const cv::Mat& image) const
{
    if (image.empty()) { throw std::runtime_error("Cannot convert an empty image to grayscale."); }
    if (image.channels() == 1) { return image; }

    cv::Mat grayscaleImage;
    cv::cvtColor(image, grayscaleImage, cv::COLOR_BGR2GRAY);

    return grayscaleImage;
}

DenseMatchingResult DenseStereoMatcher::BuildResult(const cv::Mat& rawDisparity, const DenseStereoConfig& config) const
{
    if (rawDisparity.empty()) { throw std::runtime_error("Cannot build result from an empty disparity map."); }
    DenseMatchingResult result;

    /*
        OpenCV stores BM and SGBM disparities as fixed-point values
        scaled by 16. Convert them to actual pixel disparities.
    */
    rawDisparity.convertTo(result.rawDisparity, CV_32F, 1.0 / 16.0);

    /*
        Median blur removes isolated salt-and-pepper disparities cheaply. The
        invalid sentinel (~minDisparity - 1) is far from real values, so a small
        kernel does not pull valid pixels toward it noticeably.
    */
    if (config.openCv.medianKernel >= 3 && config.openCv.medianKernel % 2 == 1)
    {
        cv::medianBlur(result.rawDisparity, result.rawDisparity, config.openCv.medianKernel);
    }

    result.validDisparityMask = result.rawDisparity > static_cast<float>(config.openCv.minDisparity);

    // Diagnostic: detect when numDisparities clips the scene (real disparities
    // piling up at the upper search bound -> raise numDisparities to recover them).
    double minVal = 0.0;
    double maxVal = 0.0;
    cv::minMaxLoc(result.rawDisparity, &minVal, &maxVal, nullptr, nullptr, result.validDisparityMask);
    const int searchTop = config.openCv.minDisparity + config.openCv.numDisparities;
    std::cout << "Disparity range observed: [" << minVal << ", " << maxVal << "] px (search window up to "
        << searchTop << ").";
    if (maxVal >= searchTop - 1)
    {
        std::cout << " WARNING: disparities reach the search bound; consider increasing numDisparities.";
    }
    std::cout << std::endl;

    cv::normalize(result.rawDisparity, result.disparityVisualization, 0, 255, cv::NORM_MINMAX, CV_8U, result.validDisparityMask);

    return result;
}

cv::Mat DenseStereoMatcher::ComputeSgbmDisparity(const cv::Mat& leftImage, const cv::Mat& rightImage, const cv::Mat& leftGuide, const DenseStereoConfig& config, cv::Mat& confidenceMapOut) const
{
	confidenceMapOut.release();
    const int p1 = config.openCv.p1 > 0 ? config.openCv.p1 : 8 * config.openCv.blockSize * config.openCv.blockSize;
    const int p2 = config.openCv.p2 > 0 ? config.openCv.p2 : 32 * config.openCv.blockSize * config.openCv.blockSize;

    cv::Ptr<cv::StereoSGBM> matcher = cv::StereoSGBM::create(config.openCv.minDisparity, config.openCv.numDisparities, config.openCv.blockSize, p1, p2);
    matcher->setUniquenessRatio(config.openCv.uniquenessRatio);
    matcher->setSpeckleWindowSize(config.openCv.speckleWindowSize);
    matcher->setSpeckleRange(config.openCv.speckleRange);
    matcher->setDisp12MaxDiff(config.openCv.disp12MaxDiff);
    matcher->setPreFilterCap(config.openCv.preFilterCap);
    matcher->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);

#ifdef HAVE_OPENCV_XIMGPROC
    if (config.openCv.useWlsFilter)
    {
        /*
            Weighted-least-squares filtering: smooth the disparity while snapping
            to color edges in the left view, using a right-view disparity for a
            left-right confidence map. This is the single biggest noise reducer
            for the classical pipeline. Operates on the raw fixed-point (CV_16S)
            disparities, so run it before the /16 conversion in BuildResult.
        */
        cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls = cv::ximgproc::createDisparityWLSFilter(matcher);
        cv::Ptr<cv::StereoMatcher> rightMatcher = cv::ximgproc::createRightMatcher(matcher);

        cv::Mat leftDisparity;
        cv::Mat rightDisparity;
        matcher->compute(leftImage, rightImage, leftDisparity);
        rightMatcher->compute(rightImage, leftImage, rightDisparity);

        wls->setLambda(config.openCv.wlsLambda);
        wls->setSigmaColor(config.openCv.wlsSigma);

        cv::Mat filteredDisparity;
        wls->filter(leftDisparity, leftGuide, filteredDisparity, rightDisparity);
        confidenceMapOut = wls->getConfidenceMap(); // CV_32F, 0..255

        std::cout << "WLS disparity filtering applied (lambda=" << config.openCv.wlsLambda << ", sigma=" << config.openCv.wlsSigma << ")." << std::endl;
        return filteredDisparity;
    }
#else
    (void)leftGuide;
    if (config.openCv.useWlsFilter)
    {
        std::cout << "WLS requested but this build lacks opencv_ximgproc; using unfiltered SGBM. " << "Install opencv4[contrib] to enable it." << std::endl;
    }
#endif

    cv::Mat rawDisparity;
    matcher->compute(leftImage, rightImage, rawDisparity);
    return rawDisparity;
}

DenseMatchingResult DenseStereoMatcher::ComputeDisparity(const RectificationResult& rectificationResult, const DenseStereoConfig& config) const
{
    std::cout << "Disparity map computation started." << std::endl;

    if (config.openCv.numDisparities <= 0 || config.openCv.numDisparities % 16 != 0)
    {
        throw std::runtime_error("numDisparities must be a positive multiple of 16.");
    }

    if (config.openCv.blockSize <= 0 || config.openCv.blockSize % 2 == 0)
    {
        throw std::runtime_error("blockSize must be a positive odd number.");
    }

    const cv::Mat leftImage = ConvertToGrayscale(rectificationResult.rectifiedLeftImage);
    const cv::Mat rightImage = ConvertToGrayscale(rectificationResult.rectifiedRightImage);

    cv::Mat rawDisparity;
    cv::Mat confidenceMap;

    if (config.openCv.method == DenseStereoMethod::StereoBM)
    {
        cv::Ptr<cv::StereoBM> matcher = cv::StereoBM::create(config.openCv.numDisparities, config.openCv.blockSize);
        matcher->setMinDisparity(config.openCv.minDisparity);
        matcher->compute(leftImage, rightImage, rawDisparity);
    }
    else
    {
        // Guide WLS with the rectified left color image for sharper edge alignment.
        rawDisparity = ComputeSgbmDisparity(leftImage, rightImage, rectificationResult.rectifiedLeftImage, config, confidenceMap);
    }

    std::cout << "Disparity map computation finished successfully." << std::endl;

    DenseMatchingResult result = BuildResult(rawDisparity, config);

    /*
        Drop WLS-extrapolated pixels below the confidence threshold so the valid
        mask reflects actually-matched geometry rather than the smooth fill.
    */
    if (!confidenceMap.empty() && config.openCv.wlsConfidenceThreshold > 0.0f)
    {
        const int beforeCount = cv::countNonZero(result.validDisparityMask);
        cv::Mat confidentMask = confidenceMap >= (config.openCv.wlsConfidenceThreshold * 255.0f);
        cv::bitwise_and(result.validDisparityMask, confidentMask, result.validDisparityMask);
        const int afterCount = cv::countNonZero(result.validDisparityMask);
        std::cout << "WLS confidence gate (>= " << config.openCv.wlsConfidenceThreshold << "): kept "
            << afterCount << " / " << beforeCount << " disparities." << std::endl;
    }

    return result;
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
