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
		StereoSGBM/StereoBM (minDisparity >= 0) only search for matches to the left
		in the right image, which assumes the second camera is to the RIGHT of the
		first, i.e. the recovered baseline has t.x < 0. If t.x > 0 the loaded pair
		is in the wrong order; swap the images (and the matched points) and recompute
		the geometry so disparities are positive and the depth comes out correct.
	*/
	if (geometry.translation.x() > 0.0)
	{
		std::cout << "Recovered t.x > 0; swapping left/right to satisfy the disparity convention." << std::endl;
		std::swap(imagePair.leftImage, imagePair.rightImage);
		std::swap(imagePair.leftImgPath, imagePair.rightImgPath);
		std::swap(matchingResult.leftMatchedPoints, matchingResult.rightMatchedPoints);
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

	// Step output: colored disparity map.
	cv::Mat disparityColored;
	cv::applyColorMap(sgbmResult.disparityVisualization, disparityColored, cv::COLORMAP_JET);
	disparityColored.setTo(cv::Scalar(0, 0, 0), ~sgbmResult.validDisparityMask);
	cv::imwrite((outputDir / "disparity_sgbm.png").string(), disparityColored);

	// Final stage: disparity -> depth -> colored point cloud + mesh.
	DepthReconstructor depthReconstructor;
	DepthReconstructionConfig depthConfig;
	depthConfig.metricBaseline = 1.0; // up to scale; set to the real ETH3D baseline (m) for metric depth
	ReconstructionResult reconstruction = depthReconstructor.Reconstruct(rectResult, sgbmResult, depthConfig);
	reconstruction.PrintStats();

	if (reconstruction.ValidPointCount() > 0)
	{
		reconstruction.WritePointCloudPly(outputDir / "pointcloud_sgbm.ply");
		reconstruction.WriteMeshPly(outputDir / "mesh_sgbm.ply", depthConfig.maxMeshEdgeDepthDiff);
	}
	else
	{
		std::cout << "No valid 3D points were reconstructed; skipping point cloud and mesh export." << std::endl;
	}

	return 0;
}
