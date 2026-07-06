#include "CustomDenseMatcher.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace
{
	// Invalid-disparity sentinel. Real disparities are >= 0, so any value below
	// this marks a rejected pixel. Mirrors the "< minDisparity" convention used by
	// the OpenCV path so DepthReconstructor's mask handling stays uniform.
	constexpr float kInvalidDisparity = -1.0f;

	// Window sum (un-normalized box filter) on a CV_32F image. This is the only
	// OpenCV "matching" helper we use; it is a numerical primitive (a separable
	// running sum), not a stereo algorithm.
	cv::Mat WindowSum(const cv::Mat& src, int windowSize)
	{
		cv::Mat dst;
		cv::boxFilter(src, dst, CV_32F, cv::Size(windowSize, windowSize),
			cv::Point(-1, -1), /*normalize=*/false, cv::BORDER_REFLECT);
		return dst;
	}

	/*
		Shift the target image horizontally so that shifted(x) == target(x - d) for
		leftToRight (the left view's match lies d pixels to the left in the right
		view), or shifted(x) == target(x + d) otherwise. Out-of-range columns are
		left zero; those pixels are invalidated by the search-border strip later.
	*/
	cv::Mat ShiftHorizontally(const cv::Mat& target, int d, bool leftToRight)
	{
		cv::Mat shifted = cv::Mat::zeros(target.size(), target.type());
		if (d == 0) { target.copyTo(shifted); return shifted; }

		const int width = target.cols;
		if (d >= width) { return shifted; }

		if (leftToRight)
		{
			// shifted(x) = target(x - d): copy [0, W-d) -> [d, W)
			target.colRange(0, width - d).copyTo(shifted.colRange(d, width));
		}
		else
		{
			// shifted(x) = target(x + d): copy [d, W) -> [0, W-d)
			target.colRange(d, width).copyTo(shifted.colRange(0, width - d));
		}
		return shifted;
	}
}

cv::Mat CustomDenseMatcher::ConvertToGrayscale(const cv::Mat& image) const
{
	if (image.empty()) { throw std::runtime_error("Cannot convert an empty image to grayscale."); }
	if (image.channels() == 1) { return image; }

	cv::Mat grayscaleImage;
	cv::cvtColor(image, grayscaleImage, cv::COLOR_BGR2GRAY);
	return grayscaleImage;
}

cv::Mat CustomDenseMatcher::MatchDirection(const cv::Mat& referenceGray, const cv::Mat& targetGray, const DenseStereoConfig& config, bool leftToRight) const
{
	const int windowSize = config.customWindowSize;
	const int radius = windowSize / 2;
	const int minD = config.customMinDisparity;
	const int numD = config.customNumDisparities;
	const float windowArea = static_cast<float>(windowSize * windowSize);
	constexpr float kEps = 1e-6f;
	constexpr float kSentinelNcc = -2.0f; // below the [-1, 1] NCC range

	const cv::Size size = referenceGray.size();

	// Reference-window statistics depend only on the reference image -> compute once.
	cv::Mat refSq;
	cv::multiply(referenceGray, referenceGray, refSq);
	const cv::Mat sumRef = WindowSum(referenceGray, windowSize);
	const cv::Mat sumRefSq = WindowSum(refSq, windowSize);

	// Per-pixel winners and the NCC values at the winner's neighbours (for the
	// parabola subpixel fit). We sweep d in increasing order and remember the
	// previous d's NCC so we can capture the winner's left/right neighbours.
	cv::Mat bestNcc(size, CV_32F, cv::Scalar(kSentinelNcc));
	cv::Mat bestD(size, CV_32F, cv::Scalar(kInvalidDisparity));
	cv::Mat nccMinus(size, CV_32F, cv::Scalar(kSentinelNcc));
	cv::Mat nccPlus(size, CV_32F, cv::Scalar(kSentinelNcc));
	cv::Mat prevNcc(size, CV_32F, cv::Scalar(kSentinelNcc));

	for (int d = minD; d < minD + numD; ++d)
	{
		const cv::Mat shiftedTgt = ShiftHorizontally(targetGray, d, leftToRight);

		cv::Mat shiftedSq;
		cv::multiply(shiftedTgt, shiftedTgt, shiftedSq);
		cv::Mat product;
		cv::multiply(referenceGray, shiftedTgt, product);

		const cv::Mat sumTgt = WindowSum(shiftedTgt, windowSize);
		const cv::Mat sumTgtSq = WindowSum(shiftedSq, windowSize);
		const cv::Mat sumProd = WindowSum(product, windowSize);

		// NCC = cov(ref, tgt) / (std(ref) * std(tgt)) over the window.
		const cv::Mat meanRef = sumRef / windowArea;
		const cv::Mat meanTgt = sumTgt / windowArea;
		cv::Mat cov = sumProd / windowArea - meanRef.mul(meanTgt);
		cv::Mat varRef = sumRefSq / windowArea - meanRef.mul(meanRef);
		cv::Mat varTgt = sumTgtSq / windowArea - meanTgt.mul(meanTgt);

		cv::max(varRef, 0.0, varRef);
		cv::max(varTgt, 0.0, varTgt);
		cv::Mat denom;
		cv::sqrt(varRef.mul(varTgt), denom);

		cv::Mat ncc = cov / (denom + kEps);
		// Untextured windows (near-zero variance) give meaningless correlation.
		cv::Mat lowTexture = denom < kEps;
		ncc.setTo(kSentinelNcc, lowTexture);

		// Capture the winner's right neighbour: pixels whose current best is d-1.
		cv::Mat prevWasBest;
		cv::compare(bestD, static_cast<double>(d - 1), prevWasBest, cv::CMP_EQ);
		ncc.copyTo(nccPlus, prevWasBest);

		// Update winners where this disparity correlates better.
		cv::Mat better = ncc > bestNcc;
		ncc.copyTo(bestNcc, better);
		bestD.setTo(static_cast<double>(d), better);
		prevNcc.copyTo(nccMinus, better);            // left neighbour = NCC at d-1
		nccPlus.setTo(kSentinelNcc, better);         // right neighbour unknown yet

		ncc.copyTo(prevNcc);
	}

	// Subpixel parabola fit using the winner and its two neighbours.
	cv::Mat disparity = bestD.clone();
	if (config.customSubpixel)
	{
		for (int row = 0; row < size.height; ++row)
		{
			const float* dPtr = bestD.ptr<float>(row);
			const float* cPtr = bestNcc.ptr<float>(row);
			const float* mPtr = nccMinus.ptr<float>(row);
			const float* pPtr = nccPlus.ptr<float>(row);
			float* outPtr = disparity.ptr<float>(row);

			for (int col = 0; col < size.width; ++col)
			{
				if (dPtr[col] <= kInvalidDisparity) { continue; }
				const float c = cPtr[col];
				const float m = mPtr[col];
				const float p = pPtr[col];
				if (m <= kSentinelNcc || p <= kSentinelNcc) { continue; } // winner at search edge

				const float denomParabola = (m - 2.0f * c + p);
				if (std::abs(denomParabola) < 1e-6f) { continue; }
				const float delta = 0.5f * (m - p) / denomParabola;
				if (delta > -1.0f && delta < 1.0f) { outPtr[col] += delta; }
			}
		}
	}

	// Reject low-confidence matches.
	cv::Mat weak = bestNcc < config.customMinCorrelation;
	disparity.setTo(kInvalidDisparity, weak);

	/*
		Invalidate the search/window border: the window radius (incomplete windows)
		and the strip with no valid target overlap for the search range.
	*/
	const int searchSpan = minD + numD;
	disparity.rowRange(0, std::min(radius, size.height)).setTo(kInvalidDisparity);
	disparity.rowRange(std::max(0, size.height - radius), size.height).setTo(kInvalidDisparity);
	if (leftToRight)
	{
		disparity.colRange(0, std::min(searchSpan + radius, size.width)).setTo(kInvalidDisparity);
		disparity.colRange(std::max(0, size.width - radius), size.width).setTo(kInvalidDisparity);
	}
	else
	{
		disparity.colRange(0, std::min(radius, size.width)).setTo(kInvalidDisparity);
		disparity.colRange(std::max(0, size.width - searchSpan - radius), size.width).setTo(kInvalidDisparity);
	}

	return disparity;
}

cv::Mat CustomDenseMatcher::MatchNcc(const cv::Mat& leftGray, const cv::Mat& rightGray, const DenseStereoConfig& config) const
{
	cv::Mat disparityLeft = MatchDirection(leftGray, rightGray, config, /*leftToRight=*/true);

	if (config.customLrConsistency <= 0.0f) { return disparityLeft; }

	// Left-right consistency: re-match right->left and reject left pixels whose
	// disparity disagrees with the reverse match (occlusions / mismatches).
	const cv::Mat disparityRight = MatchDirection(rightGray, leftGray, config, /*leftToRight=*/false);

	const int width = leftGray.cols;
	int rejected = 0;
	for (int row = 0; row < leftGray.rows; ++row)
	{
		float* leftPtr = disparityLeft.ptr<float>(row);
		const float* rightPtr = disparityRight.ptr<float>(row);

		for (int col = 0; col < width; ++col)
		{
			const float dL = leftPtr[col];
			if (dL <= kInvalidDisparity) { continue; }

			const int matchCol = static_cast<int>(std::lround(col - dL));
			if (matchCol < 0 || matchCol >= width)
			{
				leftPtr[col] = kInvalidDisparity;
				++rejected;
				continue;
			}

			const float dR = rightPtr[matchCol];
			if (dR <= kInvalidDisparity || std::abs(dR - dL) > config.customLrConsistency)
			{
				leftPtr[col] = kInvalidDisparity;
				++rejected;
			}
		}
	}

	std::cout << "Custom matcher: left-right consistency rejected " << rejected << " pixels." << std::endl;
	return disparityLeft;
}

DenseMatchingResult CustomDenseMatcher::ComputeDisparity(const RectificationResult& rectificationResult, const DenseStereoConfig& config) const
{
	std::cout << "Custom NCC disparity computation started." << std::endl;

	if (config.customWindowSize <= 0 || config.customWindowSize % 2 == 0)
	{
		throw std::runtime_error("customWindowSize must be a positive odd number.");
	}
	if (config.customNumDisparities <= 0)
	{
		throw std::runtime_error("customNumDisparities must be positive.");
	}
	if (config.customDownscale <= 0.0 || config.customDownscale > 1.0)
	{
		throw std::runtime_error("customDownscale must be in (0, 1].");
	}

	const cv::Mat leftFullGray = ConvertToGrayscale(rectificationResult.rectifiedLeftImage);
	const cv::Mat rightFullGray = ConvertToGrayscale(rectificationResult.rectifiedRightImage);

	cv::Mat leftFloat;
	cv::Mat rightFloat;
	leftFullGray.convertTo(leftFloat, CV_32F);
	rightFullGray.convertTo(rightFloat, CV_32F);

	// Run on a downscaled pair for tractable runtime.
	const double ds = config.customDownscale;
	cv::Mat leftWork;
	cv::Mat rightWork;
	if (ds < 1.0)
	{
		cv::resize(leftFloat, leftWork, cv::Size(), ds, ds, cv::INTER_AREA);
		cv::resize(rightFloat, rightWork, cv::Size(), ds, ds, cv::INTER_AREA);
	}
	else
	{
		leftWork = leftFloat;
		rightWork = rightFloat;
	}

	std::cout << "Custom matcher working resolution: " << leftWork.cols << "x" << leftWork.rows
		<< " (downscale " << ds << "), window " << config.customWindowSize
		<< ", disparities " << config.customMinDisparity << ".."
		<< (config.customMinDisparity + config.customNumDisparities - 1) << std::endl;

	const cv::Mat disparityWork = MatchNcc(leftWork, rightWork, config);

	// Build a working-resolution validity mask, then upscale both to full size.
	// Nearest-neighbour upscaling keeps the invalid sentinel distinct (no
	// interpolation of valid into invalid).
	cv::Mat maskWork = disparityWork > kInvalidDisparity;

	const cv::Size fullSize = leftFullGray.size();
	cv::Mat disparityFull;
	cv::Mat maskFull;
	cv::resize(disparityWork, disparityFull, fullSize, 0, 0, cv::INTER_NEAREST);
	cv::resize(maskWork, maskFull, fullSize, 0, 0, cv::INTER_NEAREST);

	// Disparity scales with image width: a disparity of d at the downscaled
	// resolution corresponds to d / downscale full-resolution pixels.
	if (ds < 1.0)
	{
		disparityFull *= static_cast<float>(1.0 / ds);
	}
	disparityFull.setTo(kInvalidDisparity, ~maskFull);

	DenseMatchingResult result;
	result.rawDisparity = disparityFull;
	result.validDisparityMask = maskFull;
	cv::normalize(result.rawDisparity, result.disparityVisualization, 0, 255, cv::NORM_MINMAX, CV_8U, result.validDisparityMask);

	double minVal = 0.0;
	double maxVal = 0.0;
	cv::minMaxLoc(result.rawDisparity, &minVal, &maxVal, nullptr, nullptr, result.validDisparityMask);
	std::cout << "Custom disparity range observed: [" << minVal << ", " << maxVal << "] full-res px. Kept "
		<< cv::countNonZero(result.validDisparityMask) << " / " << (fullSize.width * fullSize.height)
		<< " pixels." << std::endl;
	std::cout << "Custom NCC disparity computation finished successfully." << std::endl;

	return result;
}
