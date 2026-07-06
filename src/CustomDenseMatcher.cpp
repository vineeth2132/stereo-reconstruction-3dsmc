#include "CustomDenseMatcher.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace
{
	// Invalid-disparity sentinel. Real disparities are >= 0, so any value below
	// this marks a rejected pixel. Mirrors the "< minDisparity" convention used by
	// the OpenCV path so DepthReconstructor's mask handling stays uniform.
	constexpr float kInvalidDisparity = -1.0f;
	constexpr float kSentinelNcc = -2.0f; // below the [-1, 1] NCC range

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
		Shift the image horizontally so that shifted(x) == image(x - d) for
		leftToRight (the left view's match lies d pixels to the left in the right
		view), or shifted(x) == image(x + d) otherwise. d may be negative (used by
		the residual sweep, which searches on both sides of the prior). Out-of-range
		columns are left zero; those pixels are invalidated by the border strips /
		out-of-image masks later.
	*/
	cv::Mat ShiftHorizontally(const cv::Mat& image, int d, bool leftToRight)
	{
		cv::Mat shifted = cv::Mat::zeros(image.size(), image.type());
		// Net rightward shift of the content: shifted(x) = image(x - shift).
		const int shift = leftToRight ? d : -d;
		if (shift == 0) { image.copyTo(shifted); return shifted; }

		const int width = image.cols;
		if (std::abs(shift) >= width) { return shifted; }

		if (shift > 0)
		{
			// shifted(x) = image(x - shift): copy [0, W-shift) -> [shift, W)
			image.colRange(0, width - shift).copyTo(shifted.colRange(shift, width));
		}
		else
		{
			// shifted(x) = image(x + |shift|): copy [|shift|, W) -> [0, W-|shift|)
			const int a = -shift;
			image.colRange(a, width).copyTo(shifted.colRange(0, width - a));
		}
		return shifted;
	}

	/*
		Vectorized global-shift NCC sweep of `target` against `reference` over the
		integer disparity hypotheses s in [sLo, sHi] (in the leftToRight sign
		convention). Returns a CV_32F disparity holding the winning s (+ parabola
		subpixel delta if requested); pixels whose best correlation is below
		minCorrelation, or that sit inside the incomplete-window border, get the
		invalid sentinel. This is the shared workhorse for both the coarse full
		search and the per-level residual search.
	*/
	cv::Mat NccSweep(const cv::Mat& reference, const cv::Mat& target, int sLo, int sHi,
		int windowSize, float minCorrelation, bool subpixel, bool leftToRight)
	{
		const int radius = windowSize / 2;
		const float windowArea = static_cast<float>(windowSize * windowSize);
		constexpr float kEps = 1e-6f;

		const cv::Size size = reference.size();

		// Reference-window statistics depend only on the reference image -> compute once.
		cv::Mat refSq;
		cv::multiply(reference, reference, refSq);
		const cv::Mat sumRef = WindowSum(reference, windowSize);
		const cv::Mat sumRefSq = WindowSum(refSq, windowSize);

		// Per-pixel winners and the NCC values at the winner's neighbours (for the
		// parabola subpixel fit). We sweep s in increasing order (step 1) and
		// remember the previous s's NCC so we can capture the winner's left/right
		// neighbours.
		cv::Mat bestNcc(size, CV_32F, cv::Scalar(kSentinelNcc));
		cv::Mat bestD(size, CV_32F, cv::Scalar(kInvalidDisparity));
		cv::Mat nccMinus(size, CV_32F, cv::Scalar(kSentinelNcc));
		cv::Mat nccPlus(size, CV_32F, cv::Scalar(kSentinelNcc));
		cv::Mat prevNcc(size, CV_32F, cv::Scalar(kSentinelNcc));

		for (int d = sLo; d <= sHi; ++d)
		{
			const cv::Mat shiftedTgt = ShiftHorizontally(target, d, leftToRight);

			cv::Mat shiftedSq;
			cv::multiply(shiftedTgt, shiftedTgt, shiftedSq);
			cv::Mat product;
			cv::multiply(reference, shiftedTgt, product);

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
		if (subpixel)
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
		cv::Mat weak = bestNcc < minCorrelation;
		disparity.setTo(kInvalidDisparity, weak);

		// Invalidate the incomplete-window border (radius on every side). The extra
		// search-span / out-of-image invalidation is applied by the caller, which
		// knows the level's search geometry.
		disparity.rowRange(0, std::min(radius, size.height)).setTo(kInvalidDisparity);
		disparity.rowRange(std::max(0, size.height - radius), size.height).setTo(kInvalidDisparity);
		disparity.colRange(0, std::min(radius, size.width)).setTo(kInvalidDisparity);
		disparity.colRange(std::max(0, size.width - radius), size.width).setTo(kInvalidDisparity);

		return disparity;
	}

	/*
		Fill invalid disparities by normalized convolution so every pixel carries a
		finite prior for the next pyramid level. Iteratively blurs the valid values
		(box filter of d*mask divided by box filter of mask) with a growing kernel,
		filling holes that gain support each pass, until none remain. The returned
		map is dense; the caller keeps the original validity mask separately (a
		hole-filled pixel gets a prior but is still "unconfirmed").
	*/
	cv::Mat HoleFillPrior(const cv::Mat& disparity, const cv::Mat& validMask)
	{
		constexpr float kEps = 1e-6f;

		cv::Mat filled;
		disparity.copyTo(filled);
		filled.setTo(0.0f, validMask == 0); // zero the holes so they do not pollute the sums

		cv::Mat mask; // 1.0 where valid, 0.0 where hole
		validMask.convertTo(mask, CV_32F, 1.0 / 255.0);

		int kernel = 3;
		for (int iteration = 0; iteration < 24; ++iteration)
		{
			if (cv::countNonZero(mask < 0.5f) == 0) { break; }

			const cv::Mat numerator = WindowSum(filled, kernel); // filled is 0 at holes -> equals sum of d*mask
			const cv::Mat denomSupport = WindowSum(mask, kernel);
			cv::Mat estimate = numerator / (denomSupport + kEps);

			// Only fill holes that now have at least one valid neighbour in reach.
			cv::Mat newlyValid = (mask < 0.5f) & (denomSupport > 0.5f);
			estimate.copyTo(filled, newlyValid);
			mask.setTo(1.0f, newlyValid);

			kernel += 2; // grow the reach so isolated holes eventually get support
		}

		return filled;
	}

	/*
		Backward-warp the target by the (dense) prior disparity so the residual
		search only has to correct the small leftover offset. map_x(x,y) = x - prior
		for a left reference, x + prior for a right reference; map_y = y. Regions
		whose source column falls outside the image are reported in outOfImage so the
		caller can invalidate them (rather than trusting the zero border).
	*/
	void WarpByPrior(const cv::Mat& target, const cv::Mat& prior, bool leftToRight, cv::Mat& warped, cv::Mat& outOfImage)
	{
		const cv::Size size = target.size();
		cv::Mat mapX(size, CV_32F);
		cv::Mat mapY(size, CV_32F);
		const float sign = leftToRight ? -1.0f : 1.0f;

		for (int row = 0; row < size.height; ++row)
		{
			const float* priorPtr = prior.ptr<float>(row);
			float* xPtr = mapX.ptr<float>(row);
			float* yPtr = mapY.ptr<float>(row);
			for (int col = 0; col < size.width; ++col)
			{
				xPtr[col] = static_cast<float>(col) + sign * priorPtr[col];
				yPtr[col] = static_cast<float>(row);
			}
		}

		cv::remap(target, warped, mapX, mapY, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));

		const cv::Mat tooLow = mapX < 0.0f;
		const cv::Mat tooHigh = mapX > static_cast<float>(size.width - 1);
		outOfImage = tooLow | tooHigh;
	}

	/*
		Valid-aware median filter (our own): for each valid pixel gather the valid
		disparities in the k x k window; if at least half the window is valid,
		replace the pixel with their median. Never writes at invalid pixels, so it
		cannot invent geometry in holes. Reads from copies so the pass is
		order-independent.
	*/
	void ValidMedianFilter(cv::Mat& disparity, const cv::Mat& validMask, int kernel)
	{
		if (kernel <= 1) { return; }

		const int radius = kernel / 2;
		const int threshold = (kernel * kernel) / 2; // "at least half the window valid"
		const cv::Mat src = disparity.clone();
		const cv::Mat mask = validMask.clone();
		const int rows = disparity.rows;
		const int cols = disparity.cols;

		std::vector<float> neighbours;
		neighbours.reserve(static_cast<size_t>(kernel) * kernel);

		for (int row = 0; row < rows; ++row)
		{
			const uchar* maskRow = mask.ptr<uchar>(row);
			float* outRow = disparity.ptr<float>(row);
			for (int col = 0; col < cols; ++col)
			{
				if (maskRow[col] == 0) { continue; } // never touch invalid pixels

				neighbours.clear();
				const int r0 = std::max(0, row - radius);
				const int r1 = std::min(rows - 1, row + radius);
				const int c0 = std::max(0, col - radius);
				const int c1 = std::min(cols - 1, col + radius);
				for (int wr = r0; wr <= r1; ++wr)
				{
					const uchar* wMask = mask.ptr<uchar>(wr);
					const float* wSrc = src.ptr<float>(wr);
					for (int wc = c0; wc <= c1; ++wc)
					{
						if (wMask[wc] != 0) { neighbours.push_back(wSrc[wc]); }
					}
				}

				if (static_cast<int>(neighbours.size()) < threshold) { continue; }
				const size_t mid = neighbours.size() / 2;
				std::nth_element(neighbours.begin(), neighbours.begin() + mid, neighbours.end());
				outRow[col] = neighbours[mid];
			}
		}
	}

	/*
		Custom speckle filter: label 4-connected components where two valid
		neighbours are joined iff |d_a - d_b| <= tolerance, then drop components
		smaller than minArea. Iterative BFS with an explicit stack (no recursion,
		these are multi-megapixel images).
	*/
	void SpeckleFilter(cv::Mat& disparity, cv::Mat& validMask, int minArea, float tolerance)
	{
		if (minArea <= 1) { return; }

		const int rows = disparity.rows;
		const int cols = disparity.cols;
		cv::Mat visited = cv::Mat::zeros(disparity.size(), CV_8U);

		std::vector<cv::Point> stack;
		std::vector<cv::Point> component;
		int removed = 0;

		for (int row = 0; row < rows; ++row)
		{
			for (int col = 0; col < cols; ++col)
			{
				if (validMask.at<uchar>(row, col) == 0 || visited.at<uchar>(row, col) != 0) { continue; }

				stack.clear();
				component.clear();
				stack.push_back(cv::Point(col, row));
				visited.at<uchar>(row, col) = 1;

				while (!stack.empty())
				{
					const cv::Point pixel = stack.back();
					stack.pop_back();
					component.push_back(pixel);
					const float dHere = disparity.at<float>(pixel.y, pixel.x);

					static const int dx[4] = { 1, -1, 0, 0 };
					static const int dy[4] = { 0, 0, 1, -1 };
					for (int k = 0; k < 4; ++k)
					{
						const int nx = pixel.x + dx[k];
						const int ny = pixel.y + dy[k];
						if (nx < 0 || nx >= cols || ny < 0 || ny >= rows) { continue; }
						if (validMask.at<uchar>(ny, nx) == 0 || visited.at<uchar>(ny, nx) != 0) { continue; }
						if (std::abs(disparity.at<float>(ny, nx) - dHere) > tolerance) { continue; }
						visited.at<uchar>(ny, nx) = 1;
						stack.push_back(cv::Point(nx, ny));
					}
				}

				if (static_cast<int>(component.size()) < minArea)
				{
					for (const cv::Point& pixel : component)
					{
						disparity.at<float>(pixel.y, pixel.x) = kInvalidDisparity;
						validMask.at<uchar>(pixel.y, pixel.x) = 0;
						++removed;
					}
				}
			}
		}

		std::cout << "Custom matcher: speckle filter removed " << removed << " pixels." << std::endl;
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

std::vector<cv::Mat> CustomDenseMatcher::BuildPyramid(const cv::Mat& fullFloat, const std::vector<double>& scales) const
{
	std::vector<cv::Mat> pyramid;
	pyramid.reserve(scales.size());
	for (const double scale : scales)
	{
		cv::Mat level;
		if (scale >= 1.0)
		{
			fullFloat.copyTo(level);
		}
		else
		{
			cv::resize(fullFloat, level, cv::Size(), scale, scale, cv::INTER_AREA);
		}
		pyramid.push_back(level);
	}
	return pyramid;
}

cv::Mat CustomDenseMatcher::MatchDirectionPyramid(const std::vector<cv::Mat>& referencePyramid, const std::vector<cv::Mat>& targetPyramid, const std::vector<double>& scales, const DenseStereoConfig& config, bool leftToRight) const
{
	const int windowSize = config.customWindowSize;
	const int radius = windowSize / 2;
	const float minCorrelation = config.customMinCorrelation;
	const bool subpixel = config.customSubpixel;
	const int residualRadius = config.customResidualRadius;
	const char* dirTag = leftToRight ? "L->R" : "R->L";

	cv::Mat disparity; // current level disparity (this level's px), sentinel = invalid

	for (size_t level = 0; level < scales.size(); ++level)
	{
		const cv::Mat& reference = referencePyramid[level];
		const cv::Mat& target = targetPyramid[level];
		const cv::Size size = reference.size();

		if (level == 0)
		{
			// Coarsest level: unconstrained full search. The maximum disparity is a
			// fraction of the (tiny) coarsest width, so the range comes from the
			// image geometry instead of a hand-tuned customNumDisparities bound.
			const int maxD = std::max(1, static_cast<int>(std::lround(size.width * config.customMaxDisparityFraction)));
			disparity = NccSweep(reference, target, 0, maxD, windowSize, minCorrelation, subpixel, leftToRight);

			// No valid target overlap over the search span near the reference border.
			const int searchSpan = maxD + radius;
			if (leftToRight)
			{
				disparity.colRange(0, std::min(searchSpan, size.width)).setTo(kInvalidDisparity);
			}
			else
			{
				disparity.colRange(std::max(0, size.width - searchSpan), size.width).setTo(kInvalidDisparity);
			}

			const int validCount = cv::countNonZero(disparity > kInvalidDisparity);
			std::cout << "  [" << dirTag << "] level " << level << " scale " << scales[level]
				<< " (" << size.width << "x" << size.height << ") full search 0.." << maxD
				<< " valid " << (100.0 * validCount / (size.width * size.height)) << "%" << std::endl;
			continue;
		}

		// Refine level: build a dense prior from the previous (coarser) level.
		cv::Mat prevValid = disparity > kInvalidDisparity;
		cv::Mat filledPrev = HoleFillPrior(disparity, prevValid);

		cv::Mat prior;
		cv::resize(filledPrev, prior, size, 0, 0, cv::INTER_LINEAR);
		prior *= 2.0f; // disparity scales with width; each finer level doubles it

		cv::Mat warpedTarget;
		cv::Mat outOfImage;
		WarpByPrior(target, prior, leftToRight, warpedTarget, outOfImage);

		// Residual search around the prior on the warped target.
		cv::Mat residual = NccSweep(reference, warpedTarget, -residualRadius, residualRadius, windowSize, minCorrelation, subpixel, leftToRight);

		cv::Mat residualValid = residual > kInvalidDisparity;
		cv::Mat levelDisparity = prior + residual;              // valid pixels: prior + residual (+ subpixel)
		levelDisparity.setTo(kInvalidDisparity, ~residualValid); // failed residual gate
		levelDisparity.setTo(kInvalidDisparity, outOfImage);     // warp source outside the image
		disparity = levelDisparity;

		const int validCount = cv::countNonZero(disparity > kInvalidDisparity);
		std::cout << "  [" << dirTag << "] level " << level << " scale " << scales[level]
			<< " (" << size.width << "x" << size.height << ") residual +/-" << residualRadius
			<< " valid " << (100.0 * validCount / (size.width * size.height)) << "%" << std::endl;
	}

	return disparity;
}

DenseMatchingResult CustomDenseMatcher::ComputeDisparity(const RectificationResult& rectificationResult, const DenseStereoConfig& config) const
{
	std::cout << "Custom hierarchical NCC disparity computation started." << std::endl;

	if (config.customWindowSize <= 0 || config.customWindowSize % 2 == 0)
	{
		throw std::runtime_error("customWindowSize must be a positive odd number.");
	}
	if (config.customFinalDownscale <= 0.0 || config.customFinalDownscale > 1.0)
	{
		throw std::runtime_error("customFinalDownscale must be in (0, 1].");
	}
	if (config.customCoarsestDownscale <= 0.0 || config.customCoarsestDownscale > config.customFinalDownscale)
	{
		throw std::runtime_error("customCoarsestDownscale must be in (0, customFinalDownscale].");
	}
	if (config.customResidualRadius <= 0)
	{
		throw std::runtime_error("customResidualRadius must be positive.");
	}

	const cv::Mat leftFullGray = ConvertToGrayscale(rectificationResult.rectifiedLeftImage);
	const cv::Mat rightFullGray = ConvertToGrayscale(rectificationResult.rectifiedRightImage);

	cv::Mat leftFloat;
	cv::Mat rightFloat;
	leftFullGray.convertTo(leftFloat, CV_32F);
	rightFullGray.convertTo(rightFloat, CV_32F);

	// Pyramid scales, ordered coarsest -> finest. Built by halving from the finest
	// working scale down to (roughly) the coarsest, so the coarse full search runs
	// on a tiny image and each finer level only refines a residual.
	std::vector<double> scalesFineToCoarse;
	scalesFineToCoarse.push_back(config.customFinalDownscale);
	while (scalesFineToCoarse.back() * 0.5 >= config.customCoarsestDownscale - 1e-9)
	{
		scalesFineToCoarse.push_back(scalesFineToCoarse.back() * 0.5);
	}
	std::vector<double> scales(scalesFineToCoarse.rbegin(), scalesFineToCoarse.rend());

	const std::vector<cv::Mat> leftPyramid = BuildPyramid(leftFloat, scales);
	const std::vector<cv::Mat> rightPyramid = BuildPyramid(rightFloat, scales);

	const double finalScale = config.customFinalDownscale;
	std::cout << "Custom matcher pyramid: " << scales.size() << " levels, scales "
		<< scales.front() << ".." << scales.back() << ", finest "
		<< leftPyramid.back().cols << "x" << leftPyramid.back().rows
		<< ", window " << config.customWindowSize << std::endl;

	// Left-reference pass (the disparity we keep).
	cv::Mat disparityWork = MatchDirectionPyramid(leftPyramid, rightPyramid, scales, config, /*leftToRight=*/true);

	// Left-right consistency: re-run the whole pyramid right->left and reject final
	// pixels whose disparity disagrees with the reverse match (occlusions / bad
	// matches). Applied only at the finest level.
	if (config.customLrConsistency > 0.0f)
	{
		const cv::Mat disparityRight = MatchDirectionPyramid(rightPyramid, leftPyramid, scales, config, /*leftToRight=*/false);

		const int width = disparityWork.cols;
		int rejected = 0;
		for (int row = 0; row < disparityWork.rows; ++row)
		{
			float* leftPtr = disparityWork.ptr<float>(row);
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
	}

	// Post-filtering at final working resolution (our own implementations).
	cv::Mat maskWork = disparityWork > kInvalidDisparity;
	ValidMedianFilter(disparityWork, maskWork, config.customMedianKernel);
	SpeckleFilter(disparityWork, maskWork, config.customSpeckleMinArea, config.customSpeckleTolerance);
	maskWork = disparityWork > kInvalidDisparity;

	// Upscale the final working disparity to full resolution. With the finest level
	// at customFinalDownscale this is only a small (e.g. 2x) nearest-neighbour step,
	// which keeps the invalid sentinel distinct (no interpolation of valid into
	// invalid).
	const cv::Size fullSize = leftFullGray.size();
	cv::Mat disparityFull;
	cv::Mat maskFull;
	cv::resize(disparityWork, disparityFull, fullSize, 0, 0, cv::INTER_NEAREST);
	cv::resize(maskWork, maskFull, fullSize, 0, 0, cv::INTER_NEAREST);

	// Disparity scales with image width: a disparity of d at the finest working
	// resolution corresponds to d / finalScale full-resolution pixels.
	if (finalScale < 1.0)
	{
		disparityFull *= static_cast<float>(1.0 / finalScale);
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
	std::cout << "Custom hierarchical NCC disparity computation finished successfully." << std::endl;

	return result;
}
