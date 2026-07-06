#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "DataLoader.h"
#include "SparseFeatureMatcher.h"
#include "StereoGeometryEstimator.h"
#include "StereoRectifier.h"
#include "DenseStereoMatcher.h"
#include "DepthReconstructor.h"
#include "DepthMapEvaluator.h"

int main()
{
	const std::filesystem::path outputDir = "out";
	std::filesystem::create_directories(outputDir);

	DataLoader dataLoader("data/delivery_area/");
	StereoImagePair imagePair = dataLoader.LoadStereoPair("images/dslr_images_undistorted/DSC_0688.jpg", "images/dslr_images_undistorted/DSC_0689.jpg");
	//imagePair.ShowResized(0.15);

	const CameraIntrinsics camIntrinsics = dataLoader.LoadCameraIntrinsics("dslr_calibration_undistorted/cameras.txt");
	//camIntrinsics.Print();

	SparseFeatureMatcher featureMatcher;
	StereoFeatureSet featureSet = featureMatcher.DetectFeatures(imagePair);
	//featureMatcher.ShowKeypoints(imagePair, featureSet);

	float ratioThreshold = 0.5f; // Decreasing the ratioThreshold decreases the number of matches, increases the confidence of matches
	SparseMatchingResult matchingResult = featureMatcher.MatchFeatures(featureSet, ratioThreshold);
	//featureMatcher.ShowMatches(imagePair, featureSet, matchingResult);

	StereoGeometryEstimator geometryEstimator;
	StereoGeometry geometry = geometryEstimator.EstimateGeometry(matchingResult, camIntrinsics);

	/*
		Ensure the image order matches OpenCV StereoBM/SGBM's positive-disparity
		convention. If the recovered baseline points the wrong way (t.x > 0),
		swap the pair and recompute the sparse stage.
	*/
	if (geometry.translation.x() > 0.0)
	{
		std::cout << "Recovered t.x > 0; swapping left/right to satisfy the disparity convention." << std::endl;
		std::swap(imagePair.leftImage, imagePair.rightImage);
		std::swap(imagePair.leftImgPath, imagePair.rightImgPath);

		featureSet = featureMatcher.DetectFeatures(imagePair);
		matchingResult = featureMatcher.MatchFeatures(featureSet, ratioThreshold);
		geometry = geometryEstimator.EstimateGeometry(matchingResult, camIntrinsics);
	}

	geometry.PrintData();

	StereoRectifier rectifier;
	RectificationResult rectResult = rectifier.Rectify(imagePair, camIntrinsics, geometry);
	//rectResult.ShowRectifiedImages();

	// Step output: rectified pair (open in an image viewer to confirm rows align).
	cv::imwrite((outputDir / "rectified_left.png").string(), rectResult.rectifiedLeftImage);
	cv::imwrite((outputDir / "rectified_right.png").string(), rectResult.rectifiedRightImage);

	/*
		Metric baseline recovered once from the ETH3D images.txt poses. Feeds the
		depth reconstruction for both backends so the depth is metric instead of
		up-to-scale.
	*/
	float metricBaseline = dataLoader.LoadMetricBaselineFromImagesTxt("dslr_calibration_undistorted/images.txt", imagePair.leftImgPath.filename().string(), imagePair.rightImgPath.filename().string());
	std::cout << "Calculated Metric Baseline = " << metricBaseline << std::endl;

	/*
		Ground-truth depth (ETH3D) for evaluation, done once independent of the
		dense backend. The raw depth is loaded at the original DSLR resolution and
		rectified into the left rectified frame. This data may not be downloaded
		yet, so guard the whole block and continue without it if anything fails.
	*/
	try
	{
		std::string gtDepthRelativePath = "ground_truth_depth/dslr_images/" + imagePair.leftImgPath.stem().string() + ".jpg";
		cv::Mat gtDepthRaw = dataLoader.LoadGroundTruthDepthMap(gtDepthRelativePath, 6048, 4032); // 6048x4032: the raw image resolution the ETH3D depth is stored at.
		if (gtDepthRaw.empty())
		{
			throw std::runtime_error("Ground truth depth map is empty.");
		}
		cv::imwrite((outputDir / "gt_depth_raw_float.tiff").string(), gtDepthRaw);

		RawCameraCalibration rawCalibration = dataLoader.LoadRawCameraCalibration("dslr_calibration_jpg/cameras.txt");

		DepthMapEvaluator evaluator;
		cv::Mat gtDepthRectified = evaluator.RectifyGroundTruthDepth(gtDepthRaw, rawCalibration, rectResult, rectResult.rectifiedLeftImage.size());
		cv::imwrite((outputDir / "gt_depth_rectified_float.tiff").string(), gtDepthRectified);
	}
	catch (const std::exception& error)
	{
		std::cout << "Warning: skipping ground-truth depth evaluation (" << error.what()
			<< "). Download the ground_truth_depth data to enable it." << std::endl;
	}

	/*
		Dense stage, run for every backend so the OpenCV and our own custom matcher
		can be compared directly. Both implement IDenseStereoMatcher (selected by
		config.backend) and feed the identical depth/mesh tail, so the only thing
		that differs between the tagged outputs is the disparity algorithm. The
		Python helper scripts/visualize_maps.py renders the tagged maps with a
		color scale (white = invalid) for the report comparison.
	*/
	struct BackendRun
	{
		DenseStereoBackend backend;
		std::string tag;
		const char* label;
	};

	const std::vector<BackendRun> runs = {
		{ DenseStereoBackend::OpenCv, "opencv", "OpenCV StereoSGBM (+WLS)" },
		{ DenseStereoBackend::Custom, "custom", "Custom NCC block matcher" },
	};

	DepthReconstructor depthReconstructor;

	for (const BackendRun& run : runs)
	{
		std::cout << "\n==================== Dense backend: " << run.label
			<< " (tag '" << run.tag << "') ====================" << std::endl;

		DenseStereoConfig denseConfig;
		denseConfig.backend = run.backend;
		denseConfig.method = DenseStereoMethod::StereoSGBM; // used only by the OpenCV backend

		std::unique_ptr<IDenseStereoMatcher> matcher = CreateDenseMatcher(run.backend);
		DenseMatchingResult denseResult = matcher->ComputeDisparity(rectResult, denseConfig);

		// Disparity outputs (tagged per backend).
		cv::imwrite((outputDir / ("disparity_" + run.tag + "_float.tiff")).string(), denseResult.rawDisparity);
		cv::imwrite((outputDir / ("valid_disparity_" + run.tag + "_mask.tiff")).string(), denseResult.validDisparityMask);

		cv::Mat disparityColored;
		cv::applyColorMap(denseResult.disparityVisualization, disparityColored, cv::COLORMAP_JET);
		disparityColored.setTo(cv::Scalar(0, 0, 0), ~denseResult.validDisparityMask);
		cv::imwrite((outputDir / ("disparity_" + run.tag + "_colored.png")).string(), disparityColored);

		// Depth -> colored point cloud + mesh (identical tail for both backends).
		DepthReconstructionConfig depthConfig;
		depthConfig.metricBaseline = metricBaseline; // metric baseline recovered from the ETH3D poses
		// maxDepth left at 0: DepthReconstructor clips far outliers at maxDepthPercentile
		// (scene-adaptive). Set depthConfig.maxDepth explicitly to override.

		ReconstructionResult reconstruction = depthReconstructor.Reconstruct(rectResult, denseResult, depthConfig);
		reconstruction.WriteDepthMapTiff(outputDir, run.tag);
		reconstruction.PrintStats();

		if (reconstruction.ValidPointCount() > 0)
		{
			reconstruction.WritePointCloudPly(outputDir / ("pointcloud_" + run.tag + ".ply"));
			reconstruction.WriteMeshPly(outputDir / ("mesh_" + run.tag + ".ply"), depthConfig.maxMeshEdgeDepthDiff);
		}
		else
		{
			std::cout << "No valid 3D points were reconstructed for backend '" << run.tag
				<< "'; skipping point cloud and mesh export." << std::endl;
		}
	}

	return 0;
}
