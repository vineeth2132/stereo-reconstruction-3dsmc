#include "CustomDenseMatcher.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>
#include <limits>
#include <intrin.h>

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
		Sweep the target image over all integer disparity hypotheses in [sLo, sHi]
		and evaluate each candidate using normalized mean squared intensity difference over a local window.

		For every pixel, the disparity with the minimum SSD cost is selected.
		The neighbouring costs around the winner are retained for optional parabola-
		based subpixel refinement. Matches whose minimum cost exceeds maxSsdCost,
		or whose support window intersects the image border, are rejected.
	*/
	cv::Mat SsdSweep(const cv::Mat& reference, const cv::Mat& target, int sLo, int sHi, int windowSize, float maxSsdCost, float minUniqueness, bool subpixel, bool leftToRight)
	{
		const int radius = windowSize / 2;
		const cv::Size size = reference.size();

		cv::Mat bestCost(size, CV_32F, cv::Scalar(std::numeric_limits<float>::max()));
		cv::Mat bestD(size, CV_32F, cv::Scalar(kInvalidDisparity));
		cv::Mat secondBestCost(size, CV_32F, cv::Scalar(std::numeric_limits<float>::max()));
		cv::Mat costMinus(size, CV_32F, cv::Scalar(std::numeric_limits<float>::max()));
		cv::Mat costPlus(size, CV_32F, cv::Scalar(std::numeric_limits<float>::max()));
		cv::Mat previousCost(size, CV_32F, cv::Scalar(std::numeric_limits<float>::max()));

		/*
			Convert the summed squared intensity difference into a normalized
			mean squared error.

			Input intensities are currently in [0, 255], so division by
			windowArea * 255^2 maps the SSD cost approximately into [0, 1].
		*/
		constexpr float kMaxIntensity = 255.0f;
		const float normalization = static_cast<float>(windowSize * windowSize) * kMaxIntensity * kMaxIntensity;

		for (int d = sLo; d <= sHi; ++d)
		{
			const cv::Mat shiftedTarget =
				ShiftHorizontally(target, d, leftToRight);

			cv::Mat difference;
			cv::subtract(reference, shiftedTarget, difference);

			cv::Mat squaredDifference;
			cv::multiply(difference, difference, squaredDifference);

			cv::Mat cost = WindowSum(squaredDifference, windowSize);

			cost /= normalization;

			cv::Mat previousWasBest;
			cv::compare(bestD, static_cast<double>(d - 1), previousWasBest, cv::CMP_EQ);

			cost.copyTo(costPlus, previousWasBest);

			/*
				Update the winning disparity and retain the neighbouring costs required
				for the later parabolic subpixel refinement.
			*/
			cv::Mat better = cost < bestCost;

			cost.copyTo(bestCost, better);
			bestD.setTo(static_cast<double>(d), better);
			previousCost.copyTo(costMinus, better);
			costPlus.setTo(std::numeric_limits<float>::max(), better);

			cost.copyTo(previousCost);
		}

		/*
			Find the best competing disparity while excluding the winner and its
			immediate neighbours.

			The disparities bestD - 1, bestD and bestD + 1 belong to the same local
			cost minimum and are therefore not treated as independent alternatives.
		*/
		for (int d = sLo; d <= sHi; ++d)
		{
			const cv::Mat shiftedTarget = ShiftHorizontally(target, d, leftToRight);

			cv::Mat difference;
			cv::subtract(reference, shiftedTarget, difference);

			cv::Mat squaredDifference;
			cv::multiply(difference, difference, squaredDifference);

			cv::Mat cost = WindowSum(squaredDifference, windowSize);

			cost /= normalization;

			/*
				Only consider candidates farther than one disparity step from the
				winning integer disparity.
			*/
			cv::Mat distanceFromWinner;
			cv::absdiff(bestD, cv::Scalar(static_cast<float>(d)), distanceFromWinner);

			cv::Mat outsideWinnerNeighbourhood = distanceFromWinner > 1.0f;

			/*
				Ignore pixels for which the first sweep did not find any valid winner.
			*/
			cv::Mat hasWinner = bestD > kInvalidDisparity;

			cv::Mat validAlternative;
			cv::bitwise_and(outsideWinnerNeighbourhood, hasWinner, validAlternative);

			cv::Mat betterSecond = cost < secondBestCost;

			cv::bitwise_and(betterSecond, validAlternative, betterSecond);

			cost.copyTo(secondBestCost, betterSecond);
		}

		cv::Mat disparity = bestD.clone();

		if (subpixel)
		{
			for (int row = 0; row < size.height; ++row)
			{
				const float* dPtr = bestD.ptr<float>(row);
				const float* cPtr = bestCost.ptr<float>(row);
				const float* mPtr = costMinus.ptr<float>(row);
				const float* pPtr = costPlus.ptr<float>(row);
				float* outPtr = disparity.ptr<float>(row);

				for (int col = 0; col < size.width; ++col)
				{
					if (dPtr[col] <= kInvalidDisparity)
					{
						continue;
					}

					const float m = mPtr[col];
					const float c = cPtr[col];
					const float p = pPtr[col];

					if (!std::isfinite(m) || !std::isfinite(p) || !std::isfinite(c))
						continue;

					const float denominator = m - 2.0f * c + p;

					if (std::abs(denominator) < 1e-6f)
						continue;

					const float delta = 0.5f * (m - p) / denominator;

					if (delta > -1.0f && delta < 1.0f)
						outPtr[col] = dPtr[col] + delta;
				}
			}
		}

		disparity.rowRange(0, std::min(radius, size.height)).setTo(kInvalidDisparity);
		disparity.rowRange(std::max(0, size.height - radius), size.height).setTo(kInvalidDisparity);
		disparity.colRange(0, std::min(radius, size.width)).setTo(kInvalidDisparity);
		disparity.colRange(std::max(0, size.width - radius), size.width).setTo(kInvalidDisparity);

		cv::Mat weakMatches = bestCost > maxSsdCost;

		/*
			Relative separation between the best and second-best costs.

			uniqueness = 0:
				best and second-best candidates are equally good.

			Larger values:
				the winner is more clearly separated from the alternatives.
		*/
		constexpr float kUniquenessEpsilon = 1e-6f;

		cv::Mat uniqueness =
			(secondBestCost - bestCost) /
			(secondBestCost + kUniquenessEpsilon);

		cv::Mat ambiguousMatches = uniqueness < minUniqueness;

		/*
			If secondBestCost is still infinite, there was no valid second candidate.
			Do not reject such a pixel based on uniqueness alone.
		*/
		cv::Mat finiteSecondBest;
		cv::compare(secondBestCost, std::numeric_limits<float>::max(), finiteSecondBest, cv::CMP_LT);

		cv::bitwise_and(ambiguousMatches, finiteSecondBest, ambiguousMatches);

		cv::Mat rejectedMatches;
		cv::bitwise_or(weakMatches, ambiguousMatches, rejectedMatches);

		disparity.setTo(kInvalidDisparity, rejectedMatches);

		return disparity;
	}


	/*
		Compute a Census descriptor for every pixel.

		Each neighbour inside the census window is compared with the centre pixel.
		The binary comparison results are packed into a 32-bit descriptor. With the
		current radius of 2, the 5x5 neighbourhood produces 24 comparison bits.

		Pixels whose Census window crosses the image border keep the zero descriptor
		and are excluded later by CensusSweep.
	*/
	cv::Mat ComputeCensusTransform(const cv::Mat& image, int radius)
	{
		cv::Mat census(image.size(), CV_32S, cv::Scalar(0));

		for (int row = radius; row < image.rows - radius; ++row)
		{
			for (int col = radius; col < image.cols - radius; ++col)
			{
				const float center = image.at<float>(row, col);
				unsigned int descriptor = 0;

				for (int dy = -radius; dy <= radius; ++dy)
				{
					for (int dx = -radius; dx <= radius; ++dx)
					{
						if (dx == 0 && dy == 0)
							continue;

						descriptor <<= 1;

						if (image.at<float>(row + dy, col + dx) < center)
							descriptor |= 1u;
					}
				}

				census.at<int>(row, col) = static_cast<int>(descriptor);
			}
		}

		return census;
	}

	/*
		Return the Hamming distance between two Census descriptors by counting the
		set bits in their XOR result.
	*/
	int HammingDistance(unsigned int a, unsigned int b)
	{
		return __popcnt(a ^ b);
	}


	/*
		Sweep all disparity hypotheses and evaluate each match using Census distance.

		For each disparity, the target Census descriptors are compared with the
		reference descriptors using Hamming distance. Per-pixel distances are then
		aggregated over windowSize x windowSize neighbourhoods.

		The disparity with the minimum aggregated cost is selected. Candidates with
		incomplete descriptor support, border overlap or a final cost above
		maxCensusCost are rejected.
	*/
	cv::Mat CensusSweep(const cv::Mat& reference, const cv::Mat& target, int sLo, int sHi, int windowSize, int censusRadius, 
						float maxCensusCost, float minUniqueness, bool subpixel, bool leftToRight)
	{
		const cv::Size size = reference.size();
		const int borderRadius = censusRadius + windowSize / 2;
		const int descriptorBitCount = (2 * censusRadius + 1) * (2 * censusRadius + 1) - 1;
		const float invalidCost = std::numeric_limits<float>::max();

		const cv::Mat referenceCensus = ComputeCensusTransform(reference, censusRadius);
		const cv::Mat targetCensus = ComputeCensusTransform(target, censusRadius);

		cv::Mat bestCost(size, CV_32F, cv::Scalar(invalidCost));
		cv::Mat secondBestCost(size, CV_32F, cv::Scalar(invalidCost));
		cv::Mat bestD(size, CV_32F, cv::Scalar(kInvalidDisparity));
		cv::Mat costMinus(size, CV_32F, cv::Scalar(invalidCost));
		cv::Mat costPlus(size, CV_32F, cv::Scalar(invalidCost));
		cv::Mat previousCost(size, CV_32F, cv::Scalar(invalidCost));

		for (int d = sLo; d <= sHi; ++d)
		{
			const int shift = leftToRight ? d : -d;

			cv::Mat pixelCost(size, CV_32F, cv::Scalar(0.0f));

			cv::Mat validDescriptorMask(size, CV_8U, cv::Scalar(0));

			for (int row = censusRadius; row < size.height - censusRadius; ++row)
			{
				const int* referenceRow = referenceCensus.ptr<int>(row);

				float* costRow = pixelCost.ptr<float>(row);

				unsigned char* validRow = validDescriptorMask.ptr<unsigned char>(row);

				for (int col = censusRadius; col < size.width - censusRadius; ++col)
				{
					const int targetCol = col - shift;

					if (targetCol < censusRadius || targetCol >= size.width - censusRadius)
					{
						continue;
					}

					const unsigned int referenceDescriptor = static_cast<unsigned int>(referenceRow[col]);
					const unsigned int targetDescriptor = static_cast<unsigned int>(targetCensus.at<int>(row,targetCol));

					costRow[col] = static_cast<float>(HammingDistance(referenceDescriptor, targetDescriptor)) / static_cast<float>(descriptorBitCount);
					validRow[col] = 255;
				}
			}

			cv::Mat aggregatedCost = WindowSum(pixelCost, windowSize);

			aggregatedCost /= static_cast<float>(windowSize * windowSize);

			cv::Mat validMaskFloat;
			validDescriptorMask.convertTo(validMaskFloat, CV_32F, 1.0 / 255.0);

			cv::Mat validSupport = WindowSum(validMaskFloat, windowSize);

			cv::Mat incompleteWindow = validSupport < static_cast<float>(windowSize * windowSize);

			aggregatedCost.setTo(invalidCost, incompleteWindow);

			cv::Mat previousWasBest;
			cv::compare(bestD, static_cast<double>(d - 1), previousWasBest, cv::CMP_EQ);
			aggregatedCost.copyTo(costPlus, previousWasBest);

			cv::Mat better = aggregatedCost < bestCost;
			aggregatedCost.copyTo(bestCost, better);
			bestD.setTo(static_cast<double>(d), better);
			previousCost.copyTo(costMinus, better);
			costPlus.setTo(invalidCost, better);

			aggregatedCost.copyTo(previousCost);
		}

		if (minUniqueness > 0.0f)
		{
			for (int d = sLo; d <= sHi; ++d)
			{
				const int shift = leftToRight ? d : -d;

				cv::Mat pixelCost(size, CV_32F, cv::Scalar(0.0f));
				cv::Mat validDescriptorMask(size, CV_8U, cv::Scalar(0));

				for (int row = censusRadius; row < size.height - censusRadius; ++row)
				{
					const int* referenceRow = referenceCensus.ptr<int>(row);
					float* costRow = pixelCost.ptr<float>(row);
					unsigned char* validRow = validDescriptorMask.ptr<unsigned char>(row);

					for (int col = censusRadius; col < size.width - censusRadius; ++col)
					{
						const int targetCol = col - shift;

						if (targetCol < censusRadius || targetCol >= size.width - censusRadius)
						{
							continue;
						}

						const unsigned int referenceDescriptor = static_cast<unsigned int>(referenceRow[col]);
						const unsigned int targetDescriptor = static_cast<unsigned int>(targetCensus.at<int>(row, targetCol));

						costRow[col] = static_cast<float>(HammingDistance(referenceDescriptor, targetDescriptor)) / static_cast<float>(descriptorBitCount);
						validRow[col] = 255;
					}
				}

				cv::Mat aggregatedCost = WindowSum(pixelCost, windowSize);
				aggregatedCost /= static_cast<float>(windowSize * windowSize);

				cv::Mat validMaskFloat;
				validDescriptorMask.convertTo(validMaskFloat, CV_32F, 1.0 / 255.0);

				cv::Mat validSupport = WindowSum(validMaskFloat, windowSize);
				cv::Mat incompleteWindow = validSupport < static_cast<float>(windowSize * windowSize);
				aggregatedCost.setTo(invalidCost, incompleteWindow);

				cv::Mat distanceFromWinner;
				cv::absdiff(bestD, cv::Scalar(static_cast<float>(d)), distanceFromWinner);

				cv::Mat outsideWinnerNeighbourhood = distanceFromWinner > 1.0f;
				cv::Mat hasWinner = bestD > kInvalidDisparity;

				cv::Mat validAlternative;
				cv::bitwise_and(outsideWinnerNeighbourhood, hasWinner, validAlternative);

				cv::Mat betterSecond = aggregatedCost < secondBestCost;
				cv::bitwise_and(betterSecond, validAlternative, betterSecond);

				aggregatedCost.copyTo(secondBestCost, betterSecond);
			}
		}
		cv::Mat disparity = bestD.clone();

		if (subpixel)
		{
			for (int row = 0; row < size.height; ++row)
			{
				const float* dPtr = bestD.ptr<float>(row);
				const float* cPtr = bestCost.ptr<float>(row);
				const float* mPtr = costMinus.ptr<float>(row);
				const float* pPtr = costPlus.ptr<float>(row);
				float* outPtr = disparity.ptr<float>(row);

				for (int col = 0; col < size.width; ++col)
				{
					if (dPtr[col] <= kInvalidDisparity)
					{
						continue;
					}

					const float m = mPtr[col];
					const float c = cPtr[col];
					const float p = pPtr[col];

					if (!std::isfinite(m) || !std::isfinite(c) || !std::isfinite(p) ||
						m == invalidCost || c == invalidCost || p == invalidCost)
					{
						continue;
					}

					const float denominator = m - 2.0f * c + p;

					if (std::abs(denominator) < 1e-6f)
					{
						continue;
					}

					const float delta = 0.5f * (m - p) / denominator;

					if (delta > -1.0f && delta < 1.0f)
					{
						outPtr[col] = dPtr[col] + delta;
					}
				}
			}
		}

		cv::Mat weakMatches = bestCost > maxCensusCost;
		cv::Mat rejectedMatches = weakMatches.clone();

		if (minUniqueness > 0.0f)
		{
			constexpr float kUniquenessEpsilon = 1e-6f;

			cv::Mat uniqueness = (secondBestCost - bestCost) / (secondBestCost + kUniquenessEpsilon);
			cv::Mat ambiguousMatches = uniqueness < minUniqueness;

			cv::Mat finiteSecondBest;
			cv::compare(secondBestCost, std::numeric_limits<float>::max(), finiteSecondBest, cv::CMP_LT);

			cv::bitwise_and(ambiguousMatches, finiteSecondBest, ambiguousMatches);
			cv::bitwise_or(rejectedMatches, ambiguousMatches, rejectedMatches);
		}

		disparity.setTo(kInvalidDisparity, rejectedMatches);

		disparity.rowRange(0, std::min(borderRadius, size.height)).setTo(kInvalidDisparity);
		disparity.rowRange(std::max(0, size.height - borderRadius), size.height).setTo(kInvalidDisparity);
		disparity.colRange(0, std::min(borderRadius, size.width)).setTo(kInvalidDisparity);
		disparity.colRange(std::max(0, size.width - borderRadius), size.width).setTo(kInvalidDisparity);

		return disparity;
	}

	/*
		Dispatch one disparity sweep using the matching cost selected in the config.

		The coarse full search and every finer residual search pass through this
		function, so the surrounding hierarchical pipeline remains identical for
		NCC, SSD and Census.
	*/
	cv::Mat RunCostSweep(const cv::Mat& reference, const cv::Mat& target, int sLo, int sHi, const DenseStereoConfig& config, bool leftToRight)
	{
		switch (config.customCost.metric)
		{
		case CustomCostMetric::NCC:
			return NccSweep(
				reference,
				target,
				sLo,
				sHi,
				config.customCost.ncc.windowSize,
				config.customCost.ncc.minCorrelation,
				config.custom.subpixel,
				leftToRight);

		case CustomCostMetric::SSD:
			return SsdSweep(
				reference,
				target,
				sLo,
				sHi,
				config.customCost.ssd.windowSize,
				config.customCost.ssd.maxCost,
				config.customCost.ssd.minUniqueness,
				config.custom.subpixel,
				leftToRight);

		case CustomCostMetric::Census:
			return CensusSweep(
				reference,
				target,
				sLo,
				sHi,
				config.customCost.census.aggregationWindowSize,
				config.customCost.census.descriptorRadius,
				config.customCost.census.maxCost,
				config.customCost.census.minUniqueness,
				config.custom.subpixel,
				leftToRight);
		}

		throw std::runtime_error("Unknown custom cost metric.");
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

	/*
		For every pixel, the nearest valid disparity encountered when walking in the
		direction (stepX, stepY), computed in a single O(N) sweep. The recurrence
		R(p) = valid(p) ? d(p) : R(p + step) is evaluated by visiting p + step before
		p (loop bounds are ordered against the step), so one pass fills a whole
		scanline family. Pixels that never see a valid value hold the invalid
		sentinel. This is the linear-propagation primitive the directional fill uses.
	*/
	cv::Mat PropagateNearestValid(const cv::Mat& disparity, const cv::Mat& validMask, int stepX, int stepY)
	{
		const int rows = disparity.rows;
		const int cols = disparity.cols;
		cv::Mat nearest(disparity.size(), CV_32F, cv::Scalar(kInvalidDisparity));

		// Visit p + step before p: iterate descending along an axis whose step is
		// positive, ascending otherwise. A zero step direction may go either way.
		const int rowStart = stepY > 0 ? rows - 1 : 0;
		const int rowEnd = stepY > 0 ? -1 : rows;
		const int rowInc = stepY > 0 ? -1 : 1;
		const int colStart = stepX > 0 ? cols - 1 : 0;
		const int colEnd = stepX > 0 ? -1 : cols;
		const int colInc = stepX > 0 ? -1 : 1;

		for (int row = rowStart; row != rowEnd; row += rowInc)
		{
			const uchar* maskRow = validMask.ptr<uchar>(row);
			const float* dispRow = disparity.ptr<float>(row);
			float* outRow = nearest.ptr<float>(row);
			for (int col = colStart; col != colEnd; col += colInc)
			{
				if (maskRow[col] != 0)
				{
					outRow[col] = dispRow[col];
					continue;
				}
				const int nr = row + stepY;
				const int nc = col + stepX;
				if (nr >= 0 && nr < rows && nc >= 0 && nc < cols)
				{
					outRow[col] = nearest.at<float>(nr, nc); // p + step is already computed
				}
			}
		}
		return nearest;
	}

	/*
		Occlusion-aware hole filling (Hirschmuller-style). For each invalid pixel we
		gather the nearest valid disparity along each of the 8 scan directions; a hole
		is only filled when at least minDirections of them reach a valid value, which
		leaves genuinely unobserved border strips empty rather than extrapolated. The
		fill value depends on the pre-fill cause: LR-rejected holes are almost always
		true occlusions, where the correct disparity is the occluded BACKGROUND, so we
		take the second-smallest gathered value (Hirschmuller's rule; second-smallest,
		not smallest, is robust to one stray direction). Every other cause is a plain
		mismatch, so the median of the gathered values is the safer estimate. Filled
		pixels become valid; the reason map is left untouched (it stays a pre-fill
		record). The 8 directional sweeps are precomputed once, so a hole reads them
		instead of re-scanning, keeping the fill order-independent and O(N).
	*/
	void FillHolesDirectional(cv::Mat& disparity, cv::Mat& validMask, const cv::Mat& reasonMap, int minDirections)
	{
		const int rows = disparity.rows;
		const int cols = disparity.cols;

		// 8 directions: E, W, N, S, NE, NW, SE, SW (step = direction we look toward).
		static const int stepX[8] = { 1, -1, 0, 0, 1, -1, 1, -1 };
		static const int stepY[8] = { 0, 0, -1, 1, -1, -1, 1, 1 };

		cv::Mat directional[8];
		for (int d = 0; d < 8; ++d)
		{
			directional[d] = PropagateNearestValid(disparity, validMask, stepX[d], stepY[d]);
		}

		std::vector<float> gathered;
		gathered.reserve(8);
		int filledBackground = 0; // second-smallest rule (LR-rejected occlusions)
		int filledMedian = 0;     // median rule (mismatch-type holes)

		for (int row = 0; row < rows; ++row)
		{
			uchar* maskRow = validMask.ptr<uchar>(row);
			float* dispRow = disparity.ptr<float>(row);
			const uchar* reasonRow = reasonMap.ptr<uchar>(row);
			for (int col = 0; col < cols; ++col)
			{
				if (maskRow[col] != 0) { continue; } // only fill holes

				gathered.clear();
				for (int d = 0; d < 8; ++d)
				{
					const float value = directional[d].at<float>(row, col);
					if (value > kInvalidDisparity) { gathered.push_back(value); }
				}
				if (static_cast<int>(gathered.size()) < minDirections) { continue; }

				std::sort(gathered.begin(), gathered.end());
				if (reasonRow[col] == CustomMatchReasonLrRejected)
				{
					// Second-smallest = the background behind the occluding surface.
					dispRow[col] = gathered.size() >= 2 ? gathered[1] : gathered[0];
					++filledBackground;
				}
				else
				{
					dispRow[col] = gathered[gathered.size() / 2];
					++filledMedian;
				}
				maskRow[col] = 255;
			}
		}

		std::cout << "Custom matcher: directional fill filled " << (filledBackground + filledMedian)
			<< " pixels (" << filledBackground << " background/second-smallest, "
			<< filledMedian << " median)." << std::endl;
	}

	/*
		Guide-weighted diffusion of the mismatch-type fills (Jacobi normalized
		convolution), run after FillHolesDirectional and before WeightedMedianRefine.
		The directional fill plants a single value across a whole hole interior; the
		weighted median only corrects it within ~radius px of the rim, so large blob
		interiors keep a wrong constant depth. This stage relaxes those interiors toward
		a guide-weighted average of the surrounding real measurements. Matched pixels
		(reason 0) and the deliberate occlusion background (reason 3) are read-only
		ANCHORS; only mismatch fills (valid AND reason 1/2/4) are updated. Each iteration
		solves d_p <- sum_q w_q d_q / sum_q w_q over the (2r+1)^2 window, with
		w_q = exp(-(I_p - I_q)^2 / 2 sigma^2) * a_q, a_q = 1 for anchors, 0.25 for other
		fills (real measurements dominate but information still crosses hole interiors),
		0 for invalid pixels. I is the left grayscale guide at working resolution.

		The exp weight is bilinear in a quantized guide, so the whole pass reduces to box
		filters instead of a per-neighbour exp: with L intensity levels of centre c_l,
		W_l(q) = exp(-(I_q - c_l)^2 / 2 sigma^2) a_q, num_l = boxSum(W_l d) and
		den_l = boxSum(W_l); each updated pixel reads num/den linearly interpolated
		between the two levels bracketing its own intensity I_p -- exactly the brute-force
		weighted average, at box-filter cost. The weight fields W_l depend only on the
		(fixed) guide and anchor weights, so they -- and the interpolated denominator --
		are computed once and reused every iteration; only num is rebuilt as d changes.
		d is double-buffered across the window sums so the pass is Jacobi
		(order-independent), and the accumulators stay small (a handful of full-size
		buffers plus the L weight fields).
	*/
	void GuidedDiffusionRefine(cv::Mat& disparity, const cv::Mat& validMask, const cv::Mat& reasonMap, const cv::Mat& guide, int radius, float sigmaColor, int iterations)
	{
		if (iterations <= 0 || radius <= 0) { return; }

		const int rows = disparity.rows;
		const int cols = disparity.cols;
		const int windowSize = 2 * radius + 1;
		const float invTwoSigmaSq = 1.0f / (2.0f * sigmaColor * sigmaColor);
		constexpr int kLevels = 16;
		constexpr float kEps = 1e-6f;

		// Per-pixel guide weight a_q and the set of pixels this stage rewrites. Anchors
		// (matched / occlusion background) contribute at full weight and are never
		// updated; mismatch fills contribute at quarter weight and ARE updated; invalid
		// pixels are excluded (weight 0), so the sentinel never leaks into a sum.
		cv::Mat anchorWeight(disparity.size(), CV_32F, cv::Scalar(0.0f));
		cv::Mat updateMask(disparity.size(), CV_8U, cv::Scalar(0));
		for (int row = 0; row < rows; ++row)
		{
			const uchar* maskRow = validMask.ptr<uchar>(row);
			const uchar* reasonRow = reasonMap.ptr<uchar>(row);
			float* aRow = anchorWeight.ptr<float>(row);
			uchar* uRow = updateMask.ptr<uchar>(row);
			for (int col = 0; col < cols; ++col)
			{
				if (maskRow[col] == 0) { continue; }
				if (reasonRow[col] == CustomMatchReasonMatched || reasonRow[col] == CustomMatchReasonLrRejected)
				{
					aRow[col] = 1.0f; // anchor: real measurement or deliberate background
				}
				else
				{
					aRow[col] = 0.25f; // mismatch fill: both a weak source and an updated pixel
					uRow[col] = 255;
				}
			}
		}

		const int updateCount = cv::countNonZero(updateMask);
		if (updateCount == 0) { return; }

		// Quantize the guide range into kLevels centres spanning [min, max]; each pixel's
		// bracketing lower level and interpolation fraction depend only on the (fixed)
		// guide, so precompute them once.
		double guideMin = 0.0;
		double guideMax = 0.0;
		cv::minMaxLoc(guide, &guideMin, &guideMax);
		const float step = std::max(1e-3f, static_cast<float>(guideMax - guideMin) / (kLevels - 1));

		cv::Mat loIndex(disparity.size(), CV_32S);
		cv::Mat frac(disparity.size(), CV_32F);
		for (int row = 0; row < rows; ++row)
		{
			const float* gRow = guide.ptr<float>(row);
			int* loRow = loIndex.ptr<int>(row);
			float* fRow = frac.ptr<float>(row);
			for (int col = 0; col < cols; ++col)
			{
				float f = (gRow[col] - static_cast<float>(guideMin)) / step;
				int lo = static_cast<int>(std::floor(f));
				if (lo < 0) { lo = 0; }
				if (lo > kLevels - 2) { lo = kLevels - 2; }
				loRow[col] = lo;
				fRow[col] = f - static_cast<float>(lo);
			}
		}

		const double tStart = static_cast<double>(cv::getTickCount());

		// Precompute the iteration-independent per-level weight fields W_l and the
		// interpolated denominator (den_l = boxSum(W_l) also never changes, since a_q and
		// I are fixed). Only the numerator is rebuilt per iteration.
		std::vector<cv::Mat> weightLevel(kLevels);
		cv::Mat denAcc(disparity.size(), CV_32F, cv::Scalar(0.0f));
		for (int level = 0; level < kLevels; ++level)
		{
			const float centre = static_cast<float>(guideMin) + level * step;
			cv::Mat weight(disparity.size(), CV_32F);
			cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range)
			{
				for (int row = range.start; row < range.end; ++row)
				{
					const float* gRow = guide.ptr<float>(row);
					const float* aRow = anchorWeight.ptr<float>(row);
					float* wRow = weight.ptr<float>(row);
					for (int col = 0; col < cols; ++col)
					{
						const float a = aRow[col];
						if (a == 0.0f) { wRow[col] = 0.0f; continue; }
						const float diff = gRow[col] - centre;
						wRow[col] = std::exp(-(diff * diff) * invTwoSigmaSq) * a;
					}
				}
			});
			weightLevel[level] = weight;

			const cv::Mat denLevel = WindowSum(weight, windowSize);
			cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range)
			{
				for (int row = range.start; row < range.end; ++row)
				{
					const uchar* uRow = updateMask.ptr<uchar>(row);
					const int* loRow = loIndex.ptr<int>(row);
					const float* fRow = frac.ptr<float>(row);
					const float* denRow = denLevel.ptr<float>(row);
					float* denAccRow = denAcc.ptr<float>(row);
					for (int col = 0; col < cols; ++col)
					{
						if (uRow[col] == 0) { continue; }
						const int lo = loRow[col];
						if (lo == level) { denAccRow[col] += (1.0f - fRow[col]) * denRow[col]; }
						else if (lo + 1 == level) { denAccRow[col] += fRow[col] * denRow[col]; }
					}
				}
			});
		}

		for (int iter = 0; iter < iterations; ++iter)
		{
			cv::Mat numAcc(disparity.size(), CV_32F, cv::Scalar(0.0f));
			for (int level = 0; level < kLevels; ++level)
			{
				// num_l = boxSum(W_l * d) with the CURRENT disparity (invalid pixels carry
				// W_l = 0, so their sentinel contributes nothing).
				cv::Mat weightedDisp;
				cv::multiply(weightLevel[level], disparity, weightedDisp);
				const cv::Mat numLevel = WindowSum(weightedDisp, windowSize);

				cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range)
				{
					for (int row = range.start; row < range.end; ++row)
					{
						const uchar* uRow = updateMask.ptr<uchar>(row);
						const int* loRow = loIndex.ptr<int>(row);
						const float* fRow = frac.ptr<float>(row);
						const float* numRow = numLevel.ptr<float>(row);
						float* numAccRow = numAcc.ptr<float>(row);
						for (int col = 0; col < cols; ++col)
						{
							if (uRow[col] == 0) { continue; }
							const int lo = loRow[col];
							if (lo == level) { numAccRow[col] += (1.0f - fRow[col]) * numRow[col]; }
							else if (lo + 1 == level) { numAccRow[col] += fRow[col] * numRow[col]; }
						}
					}
				});
			}

			// Jacobi update: rewrite every mismatch-fill pixel from the interpolated
			// normalized convolution. Anchors are never in updateMask, so matched geometry
			// and the occlusion background are left exactly as they were.
			cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range)
			{
				for (int row = range.start; row < range.end; ++row)
				{
					const uchar* uRow = updateMask.ptr<uchar>(row);
					const float* numAccRow = numAcc.ptr<float>(row);
					const float* denAccRow = denAcc.ptr<float>(row);
					float* dRow = disparity.ptr<float>(row);
					for (int col = 0; col < cols; ++col)
					{
						if (uRow[col] == 0 || denAccRow[col] < kEps) { continue; }
						dRow[col] = numAccRow[col] / denAccRow[col];
					}
				}
			});
		}

		const double seconds = (static_cast<double>(cv::getTickCount()) - tStart) / cv::getTickFrequency();
		std::cout << "Custom matcher: guided diffusion refined " << updateCount << " pixels ("
			<< iterations << " iterations) in " << seconds << " s." << std::endl;
	}

	/*
		Edge-aware weighted median over FILLED pixels only (final-valid AND a non-zero
		pre-fill reason), guided by the left grayscale image at working resolution.
		For each filled pixel we collect its valid window neighbours with weight
		w = exp(-(I_p - I_q)^2 / (2 sigmaColor^2)) and take the weighted median (the
		value where the cumulative weight first reaches half the total). Guiding by
		intensity keeps the fill from bleeding disparity across image edges. The pass
		writes into a copy and swaps, so it is order-independent; rows are split with
		cv::parallel_for_ because the window is large over millions of filled pixels.
	*/
	void WeightedMedianRefine(cv::Mat& disparity, const cv::Mat& validMask, const cv::Mat& reasonMap, const cv::Mat& guide, int radius, float sigmaColor)
	{
		if (radius <= 0) { return; }

		const int rows = disparity.rows;
		const int cols = disparity.cols;
		const float invTwoSigmaSq = 1.0f / (2.0f * sigmaColor * sigmaColor);
		cv::Mat out = disparity.clone();

		cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range)
		{
			std::vector<std::pair<float, float>> items; // (disparity, weight)
			items.reserve(static_cast<size_t>((2 * radius + 1) * (2 * radius + 1)));

			for (int row = range.start; row < range.end; ++row)
			{
				const uchar* maskRow = validMask.ptr<uchar>(row);
				const uchar* reasonRow = reasonMap.ptr<uchar>(row);
				const float* guideRow = guide.ptr<float>(row);
				float* outRow = out.ptr<float>(row);

				for (int col = 0; col < cols; ++col)
				{
					if (maskRow[col] == 0 || reasonRow[col] == CustomMatchReasonMatched) { continue; } // filled pixels only

					const float centreIntensity = guideRow[col];
					items.clear();
					double weightSum = 0.0;

					const int r0 = std::max(0, row - radius);
					const int r1 = std::min(rows - 1, row + radius);
					const int c0 = std::max(0, col - radius);
					const int c1 = std::min(cols - 1, col + radius);
					for (int wr = r0; wr <= r1; ++wr)
					{
						const uchar* wMask = validMask.ptr<uchar>(wr);
						const float* wGuide = guide.ptr<float>(wr);
						const float* wDisp = disparity.ptr<float>(wr);
						for (int wc = c0; wc <= c1; ++wc)
						{
							if (wMask[wc] == 0) { continue; }
							const float diff = centreIntensity - wGuide[wc];
							const float weight = std::exp(-(diff * diff) * invTwoSigmaSq);
							items.emplace_back(wDisp[wc], weight);
							weightSum += weight;
						}
					}

					if (items.empty()) { continue; }

					std::sort(items.begin(), items.end(),
						[](const std::pair<float, float>& a, const std::pair<float, float>& b) { return a.first < b.first; });
					const double half = 0.5 * weightSum;
					double cumulative = 0.0;
					float median = items.back().first;
					for (const std::pair<float, float>& item : items)
					{
						cumulative += item.second;
						if (cumulative >= half) { median = item.first; break; }
					}
					outRow[col] = median;
				}
			}
		});

		disparity = out; // order-independent swap
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

/*
	Build one image pyramid from the requested scales.

	The returned levels are ordered from coarsest to finest, matching the order
	used by MatchDirectionPyramid.
*/
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

int GetMatchingWindowSize(const DenseStereoConfig& config)
{
	switch (config.customCost.metric)
	{
	case CustomCostMetric::NCC:
		return config.customCost.ncc.windowSize;

	case CustomCostMetric::SSD:
		return config.customCost.ssd.windowSize;

	case CustomCostMetric::Census:
		return config.customCost.census.aggregationWindowSize;
	}

	throw std::runtime_error("Unknown custom cost metric.");
}

cv::Mat CustomDenseMatcher::MatchDirectionPyramid(const std::vector<cv::Mat>& referencePyramid, const std::vector<cv::Mat>& targetPyramid, const std::vector<double>& scales, const DenseStereoConfig& config, bool leftToRight) const
{
	const int windowSize = GetMatchingWindowSize(config);
	const int radius = windowSize / 2;
	const int residualRadius = config.custom.residualRadius;
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
			// image geometry instead of a fixed, hand-tuned disparity count.
			const int maxD = std::max(1, static_cast<int>(std::lround(size.width * config.custom.maxDisparityFraction)));
			disparity = RunCostSweep(reference, target, 0, maxD, config, leftToRight);

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
		cv::Mat residual = RunCostSweep(reference, warpedTarget, -residualRadius, residualRadius, config, leftToRight);

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
	const char* metricName = "Unknown";

	// Resolve the selected cost name for diagnostics and progress logging.
	switch (config.customCost.metric)
	{
	case CustomCostMetric::NCC:
		metricName = "NCC";
		break;

	case CustomCostMetric::SSD:
		metricName = "SSD";
		break;

	case CustomCostMetric::Census:
		metricName = "Census";
		break;
	}

	std::cout << "Custom hierarchical " << metricName << " disparity computation started." << std::endl;

	switch (config.customCost.metric)
	{
	case CustomCostMetric::NCC:
		if (config.customCost.ncc.minCorrelation < -1.0f || config.customCost.ncc.minCorrelation > 1.0f)
		{
			throw std::runtime_error("NCC minCorrelation must be in [-1, 1].");
		}
		break;

	case CustomCostMetric::SSD:
		if (config.customCost.ssd.maxCost < 0.0f)
		{
			throw std::runtime_error("SSD maxCost must be non-negative.");
		}

		if (config.customCost.ssd.minUniqueness < 0.0f || config.customCost.ssd.minUniqueness > 1.0f)
		{
			throw std::runtime_error("SSD minUniqueness must be in [0, 1].");
		}
		break;

	case CustomCostMetric::Census:
		if (config.customCost.census.descriptorRadius < 1 || config.customCost.census.descriptorRadius > 2)
		{
			throw std::runtime_error("Census descriptorRadius must be 1 or 2 for the current 32-bit descriptor.");
		}

		if (config.customCost.census.maxCost < 0.0f)
		{
			throw std::runtime_error("Census maxCost must be non-negative.");
		}

		if (config.customCost.census.minUniqueness < 0.0f || config.customCost.census.minUniqueness > 1.0f)
		{
			throw std::runtime_error("Census minUniqueness must be in [0, 1].");
		}
		break;

	default:
		throw std::runtime_error("Unknown custom cost metric.");
	}

	const int windowSize = GetMatchingWindowSize(config);

	if (windowSize <= 0 || windowSize % 2 == 0)
	{
		throw std::runtime_error("customWindowSize must be a positive odd number.");
	}
	if (config.custom.finalDownscale <= 0.0 || config.custom.finalDownscale > 1.0)
	{
		throw std::runtime_error("customFinalDownscale must be in (0, 1].");
	}
	if (config.custom.coarsestDownscale <= 0.0 || config.custom.coarsestDownscale > config.custom.finalDownscale)
	{
		throw std::runtime_error("customCoarsestDownscale must be in (0, customFinalDownscale].");
	}
	if (config.custom.residualRadius <= 0)
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
	scalesFineToCoarse.push_back(config.custom.finalDownscale);
	while (scalesFineToCoarse.back() * 0.5 >= config.custom.coarsestDownscale - 1e-9)
	{
		scalesFineToCoarse.push_back(scalesFineToCoarse.back() * 0.5);
	}
	std::vector<double> scales(scalesFineToCoarse.rbegin(), scalesFineToCoarse.rend());

	const std::vector<cv::Mat> leftPyramid = BuildPyramid(leftFloat, scales);
	const std::vector<cv::Mat> rightPyramid = BuildPyramid(rightFloat, scales);

	const double finalScale = config.custom.finalDownscale;
	std::cout << "Custom matcher pyramid: " << scales.size() << " levels, scales "
		<< scales.front() << ".." << scales.back() << ", finest "
		<< leftPyramid.back().cols << "x" << leftPyramid.back().rows
		<< ", window " << windowSize << std::endl;

	// Left-reference pass (the disparity we keep).
	cv::Mat disparityWork = MatchDirectionPyramid(leftPyramid, rightPyramid, scales, config, /*leftToRight=*/true);

	/*
		Pre-fill failure-reason map at working resolution, seeded from the L->R
		pyramid result: every invalid pixel is "never matched" (code 1), then the
		known border / out-of-image strips are reclassified as "border" (code 2). The
		out-of-image strip is the leftmost columns up to the maximum observed
		disparity: for a left reference a pixel at column c only has a right-view
		correspondent when c >= its disparity, so the far-left band is genuinely
		unobservable rather than an unreliable match. The LR check and speckle filter
		below overwrite the pixels they reject with codes 3 and 4; whatever survives
		stays 0 (matched).
	*/
	const int reasonBorderRadius = windowSize / 2;
	cv::Mat reasonWork(disparityWork.size(), CV_8U, cv::Scalar(CustomMatchReasonMatched));
	cv::Mat invalidWork = disparityWork <= kInvalidDisparity;
	reasonWork.setTo(CustomMatchReasonNeverMatched, invalidWork);

	double maxObservedDisparity = 0.0;
	cv::minMaxLoc(disparityWork, nullptr, &maxObservedDisparity, nullptr, nullptr, ~invalidWork);
	const int leftStrip = std::min(disparityWork.cols, std::max(0, static_cast<int>(std::lround(maxObservedDisparity))));

	cv::Mat borderRegion = cv::Mat::zeros(disparityWork.size(), CV_8U);
	borderRegion.rowRange(0, std::min(reasonBorderRadius, disparityWork.rows)).setTo(255);
	borderRegion.rowRange(std::max(0, disparityWork.rows - reasonBorderRadius), disparityWork.rows).setTo(255);
	borderRegion.colRange(0, std::min(reasonBorderRadius, disparityWork.cols)).setTo(255);
	borderRegion.colRange(std::max(0, disparityWork.cols - reasonBorderRadius), disparityWork.cols).setTo(255);
	borderRegion.colRange(0, leftStrip).setTo(255); // left out-of-image band
	reasonWork.setTo(CustomMatchReasonBorder, borderRegion & invalidWork);

	// Left-right consistency: re-run the whole pyramid right->left and reject final
	// pixels whose disparity disagrees with the reverse match (occlusions / bad
	// matches). Applied only at the finest level.
	if (config.custom.lrConsistency > 0.0f)
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
					reasonWork.at<uchar>(row, col) = CustomMatchReasonLrRejected;
					++rejected;
					continue;
				}

				const float dR = rightPtr[matchCol];
				if (dR <= kInvalidDisparity || std::abs(dR - dL) > config.custom.lrConsistency)
				{
					leftPtr[col] = kInvalidDisparity;
					reasonWork.at<uchar>(row, col) = CustomMatchReasonLrRejected;
					++rejected;
				}
			}
		}
		std::cout << "Custom matcher: left-right consistency rejected " << rejected << " pixels." << std::endl;
	}

	// Post-filtering at final working resolution (our own implementations).
	cv::Mat maskWork = disparityWork > kInvalidDisparity;
	ValidMedianFilter(disparityWork, maskWork, config.custom.medianKernel);

	// Diff the mask across the speckle pass so removed pixels get cause code 4 in the
	// reason map (the median filter above never invalidates, so this mask is exact).
	const cv::Mat maskBeforeSpeckle = disparityWork > kInvalidDisparity;
	SpeckleFilter(disparityWork, maskWork, config.custom.speckleMinArea, config.custom.speckleTolerance);
	maskWork = disparityWork > kInvalidDisparity;
	reasonWork.setTo(CustomMatchReasonSpeckle, maskBeforeSpeckle & ~maskWork);

	// Snapshot the genuinely-matched mask before filling: this is the honest,
	// fill-free coverage exported for evaluation as matchedMask.
	const cv::Mat matchedMaskWork = maskWork.clone();

	/*
		Occlusion-aware densification. FillHolesDirectional turns holes with enough
		directional support into valid disparities (background rule for LR-rejected
		occlusions, median rule otherwise); WeightedMedianRefine then smooths only
		the filled pixels along image edges, guided by the finest left pyramid level
		(working resolution, CV_32F, 0..255). Both leave matched pixels and the reason
		map untouched.
	*/
	if (config.custom.fillHoles)
	{
		FillHolesDirectional(disparityWork, maskWork, reasonWork, config.custom.fillMinDirections);

		const cv::Mat& guide = leftPyramid.back(); // finest level == working resolution

		// Guide-weighted diffusion of the mismatch-type fills (reason 1/2/4) before the
		// weighted median: relaxes big blob interiors toward the surrounding real
		// measurements, which the rim-only weighted median cannot reach.
		GuidedDiffusionRefine(disparityWork, maskWork, reasonWork, guide,
			config.custom.guidedFillRadius, config.custom.guidedFillSigmaColor,
			config.custom.guidedFillIterations);

		if (config.custom.weightedMedianRadius > 0)
		{
			const double tStart = static_cast<double>(cv::getTickCount());
			for (int iteration = 0; iteration < config.custom.weightedMedianIterations; ++iteration)
			{
				WeightedMedianRefine(disparityWork, maskWork, reasonWork, guide,
					config.custom.weightedMedianRadius, config.custom.weightedMedianSigmaColor);
			}
			const double seconds = (static_cast<double>(cv::getTickCount()) - tStart) / cv::getTickFrequency();
			std::cout << "Custom matcher: weighted-median refinement (" << config.custom.weightedMedianIterations
				<< " pass(es), radius " << config.custom.weightedMedianRadius << ") took "
				<< seconds << " s." << std::endl;
		}
	}

	// Upscale the final working disparity to full resolution. With the finest level
	// at config.custom.finalDownscale this is only a small nearest-neighbour step,
	// which keeps the invalid sentinel distinct (no interpolation of valid into
	// invalid).
	const cv::Size fullSize = leftFullGray.size();
	cv::Mat disparityFull;
	cv::Mat maskFull;
	cv::resize(disparityWork, disparityFull, fullSize, 0, 0, cv::INTER_NEAREST);
	cv::resize(maskWork, maskFull, fullSize, 0, 0, cv::INTER_NEAREST);

	// Diagnostics at full resolution: the pre-fill matched mask and the pre-fill
	// reason map, nearest-upscaled to keep their discrete labels intact.
	cv::Mat matchedMaskFull;
	cv::Mat reasonFull;
	cv::resize(matchedMaskWork, matchedMaskFull, fullSize, 0, 0, cv::INTER_NEAREST);
	cv::resize(reasonWork, reasonFull, fullSize, 0, 0, cv::INTER_NEAREST);

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
	result.matchedMask = matchedMaskFull;
	result.failureReason = reasonFull;
	cv::normalize(result.rawDisparity, result.disparityVisualization, 0, 255, cv::NORM_MINMAX, CV_8U, result.validDisparityMask);

	double minVal = 0.0;
	double maxVal = 0.0;
	cv::minMaxLoc(result.rawDisparity, &minVal, &maxVal, nullptr, nullptr, result.validDisparityMask);
	std::cout << "Custom disparity range observed: [" << minVal << ", " << maxVal << "] full-res px. Kept "
		<< cv::countNonZero(result.validDisparityMask) << " / " << (fullSize.width * fullSize.height)
		<< " pixels." << std::endl;
	std::cout << "Custom hierarchical " << metricName << " disparity computation finished successfully." << std::endl;

	return result;
}
