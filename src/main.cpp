#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <optional>

#include "DataLoader.h"
#include "SparseFeatureMatcher.h"
#include "StereoGeometryEstimator.h"
#include "StereoRectifier.h"
#include "DenseStereoMatcher.h"
#include "DepthReconstructor.h"
#include "DepthMapEvaluator.h"

/*
	Strip a single trailing '/' or '\\' so the scene directory joins cleanly with
	the ETH3D-relative sub-paths regardless of whether the caller supplied a
	trailing slash.
*/
static std::filesystem::path NormalizeSceneDir(std::string sceneDir)
{
	while (!sceneDir.empty() && (sceneDir.back() == '/' || sceneDir.back() == '\\'))
	{
		sceneDir.pop_back();
	}
	return std::filesystem::path(sceneDir);
}

int main(int argc, char** argv)
{
	/*
		Scene-parameterizable entry point so the same binary can run on any ETH3D
		DSLR scene (a TA requirement). Defaults reproduce the original hardcoded
		delivery_area run exactly, so invoking with zero arguments is unchanged.

		Usage: stereo_reconstruction [sceneDir] [leftImage] [rightImage] [outDir]
		Accepted argument counts: 0 (all defaults), 1 (sceneDir), 3 (sceneDir +
		pair), 4 (all). leftImage/rightImage are relative to sceneDir.
	*/
	const char* usage =
		"Usage: stereo_reconstruction [sceneDir] [leftImage] [rightImage] [outDir]\n"
		"  sceneDir    ETH3D scene root directory (default: ../data/delivery_area/)\n"
		"  leftImage   left image path relative to sceneDir\n"
		"              (default: images/dslr_images_undistorted/DSC_0688.jpg)\n"
		"  rightImage  right image path relative to sceneDir\n"
		"              (default: images/dslr_images_undistorted/DSC_0689.jpg)\n"
		"  outDir      output directory (default: ../out)\n"
		"Accepted argument counts: 0, 1 (sceneDir), 3 (sceneDir + pair), 4 (all).\n";

	// Defaults match the original hardcoded delivery_area configuration.
	std::string sceneDirArg = "../data/delivery_area/";
	std::string leftImageRelative = "images/dslr_images_undistorted/DSC_0688.jpg";
	std::string rightImageRelative = "images/dslr_images_undistorted/DSC_0689.jpg";
	std::string outputDirArg = "../out";

	// Collect positional arguments, honoring --help / -h anywhere on the line.
	std::vector<std::string> positional;
	for (int i = 1; i < argc; ++i)
	{
		const std::string arg = argv[i];
		if (arg == "--help" || arg == "-h")
		{
			std::cout << usage;
			return 0;
		}
		positional.push_back(arg);
	}

	switch (positional.size())
	{
	case 0:
		break;
	case 1:
		sceneDirArg = positional[0];
		break;
	case 3:
		sceneDirArg = positional[0];
		leftImageRelative = positional[1];
		rightImageRelative = positional[2];
		break;
	case 4:
		sceneDirArg = positional[0];
		leftImageRelative = positional[1];
		rightImageRelative = positional[2];
		outputDirArg = positional[3];
		break;
	default:
		std::cerr << "Error: unexpected number of arguments (" << positional.size() << ").\n" << usage;
		return 1;
	}

	const std::filesystem::path sceneDir = NormalizeSceneDir(sceneDirArg);
	const std::filesystem::path outputDir = outputDirArg;

	// Echo the resolved configuration so run logs are self-documenting.
	std::cout << "Resolved configuration:" << std::endl;
	std::cout << "  Scene dir  : " << sceneDir.string() << std::endl;
	std::cout << "  Left image : " << leftImageRelative << std::endl;
	std::cout << "  Right image: " << rightImageRelative << std::endl;
	std::cout << "  Output dir : " << outputDir.string() << std::endl;

	std::filesystem::create_directories(outputDir);

	DataLoader dataLoader(sceneDir);
	StereoImagePair imagePair = dataLoader.LoadStereoPair(leftImageRelative, rightImageRelative);
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
		Rectified-frame metadata for the Python deep-stereo wrappers. The
		rectification synthesizes a new projection matrix, so its focal length
		differs from the raw cameras.txt fx; RAFT's disparity-to-depth conversion
		needs the rectified value (and it changes per scene/pair).
	*/
	{
		std::ofstream metaFile(outputDir / "rectified_meta.txt");
		metaFile << "fx " << rectResult.leftProjectionMatrix(0, 0) << "\n";
		metaFile << "baseline " << metricBaseline << "\n";
		metaFile << "width " << rectResult.rectifiedLeftImage.cols << "\n";
		metaFile << "height " << rectResult.rectifiedLeftImage.rows << "\n";
	}

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
		std::optional<CustomCostMetric> costMetric;
		std::string tag;
		const char* label;
	};

	const std::vector<BackendRun> runs = {
		{ DenseStereoBackend::OpenCv, std::nullopt, "opencv", "OpenCV StereoSGBM (+WLS)" },
		{ DenseStereoBackend::Custom, CustomCostMetric::NCC, "NCC", "Custom NCC block matcher" },
		{ DenseStereoBackend::Custom, CustomCostMetric::SSD, "SSD", "Custom SSD block matcher" },
		{ DenseStereoBackend::Custom, CustomCostMetric::Census, "Census", "Custom Census matcher" }
	};

	DepthReconstructor depthReconstructor;

	for (const BackendRun& run : runs)
	{
		std::cout << "\n==================== Dense backend: " << run.label
			<< " (tag '" << run.tag << "') ====================" << std::endl;

		DenseStereoConfig denseConfig;
		denseConfig.backend = run.backend;
		denseConfig.openCv.method = DenseStereoMethod::StereoSGBM; // used only by the OpenCV backend

		if (run.costMetric.has_value())
		{
			denseConfig.customCost.metric = run.costMetric.value();
		}

		std::unique_ptr<IDenseStereoMatcher> matcher = CreateDenseMatcher(denseConfig);
		DenseMatchingResult denseResult = matcher->ComputeDisparity(rectResult, denseConfig);

		// Depth -> colored point cloud + mesh (identical tail for both backends).
		DepthReconstructionConfig depthConfig;
		depthConfig.metricBaseline = metricBaseline; // metric baseline recovered from the ETH3D poses
		// maxDepth left at 0: DepthReconstructor clips far outliers at maxDepthPercentile
		// (scene-adaptive). Set depthConfig.maxDepth explicitly to override.
		// Export every 2nd grid point for both backends: the custom matcher works at
		// 0.5 scale so step 2 loses nothing, and it makes the meshes ~4x smaller and
		// MeshLab-safe on an old PC.
		depthConfig.exportGridStep = 2;

		if (!denseResult.failureReason.empty())
		{
			cv::imwrite((outputDir / ("reason_" + run.tag + ".png")).string(), denseResult.failureReason);
		}

		const auto writeColoredDisparity = [&](const std::string& stage, const cv::Mat& disparity, const cv::Mat& validMask)
			{
				if (disparity.empty() || validMask.empty())
				{
					return;
				}

				cv::Mat visualization;
				cv::normalize(disparity, visualization, 0, 255, cv::NORM_MINMAX, CV_8U, validMask);

				cv::Mat colored;
				cv::applyColorMap(visualization, colored, cv::COLORMAP_JET);
				colored.setTo(cv::Scalar(0, 0, 0), ~validMask);

				cv::imwrite((outputDir / ("disparity_" + run.tag + "_" + stage + "_colored.png")).string(), colored);
			};

		const auto reconstructAndExport = [&](const std::string& stage, const cv::Mat& disparity, const cv::Mat& validMask)
			{
				const std::string outputTag = run.tag + "_" + stage;

				std::cout << "\n--- Reconstruction stage: " << outputTag << " ---" << std::endl;

				ReconstructionResult reconstruction = depthReconstructor.Reconstruct(rectResult, disparity, validMask, depthConfig);

				reconstruction.WriteDepthMapTiff(outputDir, outputTag);
				reconstruction.PrintStats();

				if (reconstruction.ValidPointCount() == 0)
				{
					std::cout << "No valid 3D points for '" << outputTag << "'; skipping point cloud and mesh export." << std::endl;
					return;
				}

				reconstruction.WritePointCloudPly(outputDir / ("pointcloud_" + outputTag + ".ply"), depthConfig.exportGridStep);
				reconstruction.WriteMeshPly(outputDir / ("mesh_" + outputTag + ".ply"), depthConfig.maxMeshEdgeDepthDiff, depthConfig.exportGridStep);
			};

		if (run.backend == DenseStereoBackend::Custom)
		{
			cv::imwrite((outputDir / ("disparity_" + run.tag + "_raw_float.tiff")).string(), denseResult.rawDisparity);
			cv::imwrite((outputDir / ("valid_disparity_" + run.tag + "_raw_mask.tiff")).string(), denseResult.rawValidMask);

			cv::imwrite((outputDir / ("disparity_" + run.tag + "_filtered_float.tiff")).string(), denseResult.filteredDisparity);
			cv::imwrite((outputDir / ("valid_disparity_" + run.tag + "_filtered_mask.tiff")).string(), denseResult.filteredValidMask);

			cv::imwrite((outputDir / ("disparity_" + run.tag + "_filled_float.tiff")).string(), denseResult.filledDisparity);
			cv::imwrite((outputDir / ("valid_disparity_" + run.tag + "_filled_mask.tiff")).string(), denseResult.filledValidMask);

			writeColoredDisparity("raw", denseResult.rawDisparity, denseResult.rawValidMask);
			writeColoredDisparity("filtered", denseResult.filteredDisparity, denseResult.filteredValidMask);
			writeColoredDisparity("filled", denseResult.filledDisparity, denseResult.filledValidMask);

			reconstructAndExport("raw", denseResult.rawDisparity, denseResult.rawValidMask);
			reconstructAndExport("filtered", denseResult.filteredDisparity, denseResult.filteredValidMask);
			reconstructAndExport("filled", denseResult.filledDisparity, denseResult.filledValidMask);
		}
		else
		{
			cv::imwrite((outputDir / ("disparity_" + run.tag + "_float.tiff")).string(), denseResult.filledDisparity);
			cv::imwrite((outputDir / ("valid_disparity_" + run.tag + "_mask.tiff")).string(), denseResult.filledValidMask);

			writeColoredDisparity("final", denseResult.filledDisparity, denseResult.filledValidMask);
			reconstructAndExport("final", denseResult.filledDisparity, denseResult.filledValidMask);
		}
	}

	return 0;
}
