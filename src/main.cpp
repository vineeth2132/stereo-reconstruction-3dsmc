#include <filesystem>

#include "DataLoader.h"
#include "SparseFeatureMatcher.h"
#include "StereoGeometryEstimator.h"
#include "StereoRectifier.h"
#include "DenseStereoMatcher.h"
#include "DepthReconstructor.h"

int main()
{
	const std::filesystem::path outputDir = "../out";
	std::filesystem::create_directories(outputDir);

	DataLoader dataLoader("../data/delivery_area/");
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

	DenseStereoMatcher denseMatcher;

	DenseStereoConfig sgbmConfig;
	sgbmConfig.method = DenseStereoMethod::StereoSGBM;
	sgbmConfig.blockSize = 5;
	DenseMatchingResult sgbmResult = denseMatcher.ComputeDisparity(rectResult, sgbmConfig);
	//sgbmResult.ShowDisparity();

	// Write raw disparity map
	cv::imwrite((outputDir / "disparity_float.tiff").string(), sgbmResult.rawDisparity);
	cv::imwrite((outputDir / "valid_disparity_mask.tiff").string(), sgbmResult.validDisparityMask);

	// Step output: colored disparity map.
	cv::Mat disparityColored;
	cv::applyColorMap(sgbmResult.disparityVisualization, disparityColored, cv::COLORMAP_JET);
	disparityColored.setTo(cv::Scalar(0, 0, 0), ~sgbmResult.validDisparityMask);
	cv::imwrite((outputDir / "disparity_colored.png").string(), disparityColored);

	// Final stage: disparity -> depth -> colored point cloud + mesh.
	DepthReconstructor depthReconstructor;
	DepthReconstructionConfig depthConfig;
	depthConfig.metricBaseline = dataLoader.LoadMetricBaselineFromImagesTxt("dslr_calibration_undistorted/images.txt", imagePair.leftImgPath.filename().string(), imagePair.rightImgPath.filename().string()); // up to scale; set to the real ETH3D baseline (m) for metric depth
	std::cout << "Calculated Metric Baseline = " << depthConfig.metricBaseline << std::endl;
	// maxDepth left at 0: DepthReconstructor clips far outliers at maxDepthPercentile
	// (scene-adaptive). Set depthConfig.maxDepth explicitly to override.

	ReconstructionResult reconstruction = depthReconstructor.Reconstruct(rectResult, sgbmResult, depthConfig);
	reconstruction.WriteDepthMapTiff(outputDir);
	reconstruction.PrintStats();

	if (reconstruction.ValidPointCount() > 0)
	{
		reconstruction.WritePointCloudPly(outputDir / "pointcloud.ply");
		reconstruction.WriteMeshPly(outputDir / "mesh.ply", depthConfig.maxMeshEdgeDepthDiff);
	}
	else
	{
		std::cout << "No valid 3D points were reconstructed; skipping point cloud and mesh export." << std::endl;
	}

	return 0;
}
