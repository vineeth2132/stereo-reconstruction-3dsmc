#include "DataLoader.h"
#include "SparseFeatureMatcher.h"
#include "StereoGeometryEstimator.h"
#include "StereoRectifier.h"
#include "DenseStereoMatcher.h"

int main()
{
	DataLoader dataLoader("../data/delivery_area/");
	const StereoImagePair imagePair = dataLoader.LoadStereoPair("images/dslr_images_undistorted/DSC_0688.jpg", "images/dslr_images_undistorted/DSC_0689.jpg");
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
	//geometry.PrintData();

	StereoRectifier rectifier;
	RectificationResult rectResult = rectifier.Rectify(imagePair, camIntrinsics, geometry);
	//rectResult.ShowRectifiedImages();

	DenseStereoMatcher denseMatcher;

	//DenseStereoConfig bmConfig;
	//bmConfig.method = DenseStereoMethod::StereoBM;
	//bmConfig.blockSize = 15;
	//DenseMatchingResult bmResult = denseMatcher.ComputeDisparity(rectResult, bmConfig);
	//bmResult.ShowDisparity();

	DenseStereoConfig sgbmConfig;
	sgbmConfig.method = DenseStereoMethod::StereoSGBM;
	sgbmConfig.blockSize = 5;
	DenseMatchingResult sgbmResult = denseMatcher.ComputeDisparity(rectResult, sgbmConfig);
	sgbmResult.ShowDisparity();

	return 0;
}