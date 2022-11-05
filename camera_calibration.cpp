// -------------------------------------------------------------------------------------//
// 
//	Title: Camera Calibration and Reprojection Error Calculator
//	Author: Michael Millard (MLLMIC055)
//	Date: 17/09/2022
//	
//	Written as part of my final year research project in BSc. (Eng) in Electrical &
//	Computer Engineering at the University of Cape Town (UCT) under the supervision
//	of Associate Professor Fred Nicolls. The title of my research project is "View
//	Stitching for Machine Vision Inspection". This C++ script is concerned with 
//	extracting the intrinsic parameters, extrinsic parameters and distortion coefficients
//	of a camera and computes the root-mean-squared reprojection error (both on the set of  
//	checkerboard pattern images from which its parameter were found and on a second set of 
//	calibration images of the same checkerboard pattern taken after acquiring the required 
//	image data for my research. This was done to assess whether or not the cameras has been 
//	moved, bumped, or altered in any way during the data acquisition process and to quantify 
//	the extent of these effects.
// 
// -------------------------------------------------------------------------------------//

// -------------------------------------------------------------------------------------//
//									HEADER FILES
// -------------------------------------------------------------------------------------//

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <windows.h>
#include <fstream>
#include <stdio.h>
#include <map>

using namespace cv;
using namespace std;

// Known image size
Size imageSize = Size(1000, 800);

// -------------------------------------------------------------------------------------//
//									CAMERA CLASSES
// -------------------------------------------------------------------------------------//

// Struct attributes are public by default, class attributes are private
struct CameraParams 
{
	// Class containing attributes required for individual camera calibration
	string camName;
	Mat cameraMatrix = Mat(3, 3, CV_32F), distortionCoeffs = Mat(1, 5, CV_32F);
	vector<Mat> rVecs, tVecs;
	vector<vector<Point3f>> objectPoints;
	vector<vector<Point2f>> imagePoints;
	// Initialize reprojection errors ({} sets variables to 0)
	double reprojError1{}, reprojError2{};
};

struct StereoParams
{
	// Class containing attributes required for stereo-pair calibration
	string camNames;
	Mat cameraMatrix1 = Mat(3, 3, CV_32F), distortionCoeffs1 = Mat(1, 5, CV_32F),
		cameraMatrix2 = Mat(3, 3, CV_32F), distortionCoeffs2 = Mat(1, 5, CV_32F), 
		R = Mat(3, 3, CV_32F), T = Mat(3, 1, CV_32F),
		E = Mat(imageSize, CV_32F), F = Mat(3, 3, CV_32F);
	vector<vector<Point3f>> objectPoints;
	vector<vector<Point2f>> imagePoints1, imagePoints2;
	// Initialize reprojection error ({} sets variables to 0)
	double reprojError{};
};

// CameraParams objects used 
CameraParams cam1, cam2, cam3;
CameraParams cam1_calib2, cam2_calib2, cam3_calib2;
// StereoParams objects used
StereoParams cams21, cams32;
StereoParams cams21_calib2, cams32_calib2;

// -------------------------------------------------------------------------------------//
//									FUNCTIONS
// -------------------------------------------------------------------------------------//

double ReprojectionErrors(	const vector<vector<Point3f>> &objectPoints, const vector<vector<Point2f>> &imagePoints,
							const vector<Mat> &rVecs, const vector<Mat> &tVecs,
							const Mat &camMatrix, const Mat &distCoeffs) 
{
	/*
	This function computes the root-mean-square error (RMSE) between the observed checkerboard corner locations 
	and their predicted locations when reprojected onto the image using the camera parameters extracted.
	This operation can be done on either the same set of calibration images used to determine the camera
	parameters, or on a different set of calibration images, depending on the inputs.	
	*/

	vector<Point2f> imagePoints2;
	int totalPoints = 0;
	double totalErr = 0, err;
	vector<float> perViewErrors;
	perViewErrors.resize(objectPoints.size());

	for (int i = 0; i < (int)objectPoints.size(); ++i) 
	{
		projectPoints(Mat(objectPoints[i]), rVecs[i], tVecs[i], camMatrix, distCoeffs, imagePoints2);
		err = norm(Mat(imagePoints[i]), Mat(imagePoints2), NORM_L2);
		int n = (int)objectPoints[i].size();
		perViewErrors[i] = (float)std::sqrt(err * err / n);
		totalErr += err * err;
		totalPoints += n;
	}

	return sqrt(totalErr / totalPoints);
}

// -------------------------------------------------------------------------------------//

void CalibrateCamera(int checkerboard[2], CameraParams &cam, string imagesPath)
{
	/*
	This function takes in the dimensions (inner corners) of the checkerboard pattern used in the set of
	camera calibration images (given by the path input) and determines the intrinsic parameters, extrinsic
	parameters, distortion coefficients, and reprojection error (a RMSE) for the given CameraParams object.
	It is concerned with single camera calibration, not sterio-pair calibration.
	*/

	// Vector to store vectors of 3D points for each checkerboard image
	vector<vector<Point3f>> objPoints;
	// Vector to store vectors of 2D points for each checkerboard image
	vector<vector<Point2f>> imgPoints;
	// Defining the world coordinates for 3D points
	vector<Point3f> worldCoords;

	for (float i{ 0 }; i < checkerboard[1]; i++) {
		for (float j{ 0 }; j < checkerboard[0]; j++) {
			worldCoords.push_back(Point3f(j, i, 0.0));
		}
	}

	// Extracting path of individual image stored 
	vector<String> images;
	// Placing all image paths in a vector
	glob(imagesPath, images);
	// Creating Mat objects for images
	Mat image = Mat(imageSize, CV_32F), grey = Mat(imageSize, CV_32F);
	// Vector to store the pixel coordinates of detected checker board corners
	vector<Point2f> cornerPts;
	// Boolean for if corners are detected in the image
	bool success;

	// Looping over all the images in the directory
	for (int i{ 0 }; i < images.size(); i++) 
	{
		// Current image 
		image = imread(images[i], IMREAD_UNCHANGED);
		// Convert to greyscale (grey)
		cvtColor(image, grey, COLOR_BGR2GRAY);
		// Finding specified number of checkerboard inner corners 
		success = findChessboardCorners(grey, Size(checkerboard[0], checkerboard[1]), cornerPts,
			CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);

		if (success) 
		{
			cout << "Image " + to_string(i) + ": Success." << endl;
			// Epsilon (accuracy) down to 0.001 of a pixel or max iteration count of 30 for termination criteria
			TermCriteria criteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.001);
			// Refining pixel coordinates for given 2D points
			cornerSubPix(grey, cornerPts, Size(11, 11), Size(-1, -1), criteria);
			// Displaying the detected corner points on the checker board
			drawChessboardCorners(image, Size(checkerboard[0], checkerboard[1]), cornerPts, success);
			objPoints.push_back(worldCoords);
			imgPoints.push_back(cornerPts);
		}

		else {
			cout << "Image " + to_string(i) + ": Failed." << endl;
		}
	}

	// Display the image with the identified corner points
	imshow("Image", image);
	imwrite("./Project_Pics/Report_Pics/cvCorners.png", image);
	waitKey();

	// Save image and object points in current CameraParams object
	cam.objectPoints = objPoints;
	cam.imagePoints = imgPoints;

	// Extract camera parameters and retreive reprojection error on the calibration set of images
	double r_err = calibrateCamera(	objPoints, imgPoints, imageSize, cam.cameraMatrix,
									cam.distortionCoeffs, cam.rVecs, cam.tVecs);
	cam.reprojError1 = r_err;
	destroyAllWindows();
}	

// -------------------------------------------------------------------------------------//

void StereoCalibrate(int checkerboard[2], StereoParams& cams, string imagesPath1, string imagesPath2) 
{
	/*
	This function performs stereo-pair camera calibration using two sets of image data of the same checkerboard pattern
	captured at this same instance in time (i.e. in the same pose) from two different cameras. The intrinsic parameters
	obtained from the individual camera calibrations are used as opposed to extracting these parameters as part of the 
	stereo calibration. As such, this function simply extracts the rotation and translation parameters that define the 
	transformation from one camera's co-ordinate system to the other camera's co-ordinate system.
	*/

	// Vector to store vectors of 3D points for each checkerboard image
	vector<vector<Point3f>> objPoints;
	// Vector to store vectors of 2D points for each checkerboard image from each camera
	vector<vector<Point2f>> imgPoints1, imgPoints2;
	// Defining the world coordinates for 3D points
	vector<Point3f> worldCoords;

	for (float i{ 0 }; i < checkerboard[1]; i++) {
		for (float j{ 0 }; j < checkerboard[0]; j++) {
			worldCoords.push_back(Point3f(j, i, 0.0));
		}
	}

	// Extracting path of images of same checkerboard from each camera
	vector<String> images1, images2;
	// Placing all image paths in their respective vectors
	glob(imagesPath1, images1);
	glob(imagesPath2, images2);
	// Creating Mat objects for images
	Mat image1 = Mat(imageSize, CV_32F), image2 = Mat(imageSize, CV_32F),
		grey1 = Mat(imageSize, CV_32F), grey2 = Mat(imageSize, CV_32F);
	// Vector to store the pixel co-ordinates of detected checkerboard corners
	vector<Point2f> cornerPts1, cornerPts2;
	// Booleans for if corners are detected in each image
	boolean success1, success2;

	// Ensure image folders are the same size
	if (images1.size() != images2.size()) {
		cout << "ERROR: Number of images between folders does not match." << endl;
	}

	else
	{
		for (int i{ 0 }; i < images1.size(); i++)
		{
			// Current image 
			image1 = imread(images1[i], IMREAD_UNCHANGED);
			image2 = imread(images2[i], IMREAD_UNCHANGED);
			// Convert to greyscale (grey)
			cvtColor(image1, grey1, COLOR_BGR2GRAY);
			cvtColor(image2, grey2, COLOR_BGR2GRAY);
			// Finding specified number of checkerboard inner corners 
			success1 = findChessboardCorners(	grey1, Size(checkerboard[0], checkerboard[1]), cornerPts1,
												CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
			success2 = findChessboardCorners(	grey2, Size(checkerboard[0], checkerboard[1]), cornerPts2,
												CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);

			if (success1)
			{
				cout << "Camera 1: Image " + to_string(i) + ": Success." << endl;
				// Epsilon (accuracy) down to 0.001 of a pixel or max iteration count of 30 for termination criteria
				TermCriteria criteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.001);
				// Refining pixel coordinates for given 2D points
				cornerSubPix(grey1, cornerPts1, Size(11, 11), Size(-1, -1), criteria);
				// Displaying the detected corner points on the checker board
				drawChessboardCorners(image1, Size(checkerboard[0], checkerboard[1]), cornerPts1, success1);
				
			}
			else {
				cout << "Camera 1: Image " + to_string(i) + ": Failed." << endl;
			}

			if (success2)
			{
				cout << "Camera 2: Image " + to_string(i) + ": Success." << endl;
				// Epsilon (accuracy) down to 0.001 of a pixel or max iteration count of 30 for termination criteria
				TermCriteria criteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.001);
				// Refining pixel coordinates for given 2D points
				cornerSubPix(grey2, cornerPts2, Size(11, 11), Size(-1, -1), criteria);
				// Displaying the detected corner points on the checker board
				drawChessboardCorners(image2, Size(checkerboard[0], checkerboard[1]), cornerPts2, success2);

			}
			else {
				cout << "Camera 2: Image " + to_string(i) + ": Failed." << endl;
			}

			// Display the images with the identified corner points
			/*imshow("Image1", image1);
			imshow("Image2", image2);
			waitKey(0);*/

			if (success1 && success2)
			{
				objPoints.push_back(worldCoords);
				imgPoints1.push_back(cornerPts1);
				imgPoints2.push_back(cornerPts2);
			}
			else {
				cout << "ERROR: Failed to find chessboard corners in both frames." << endl;
			}
		}
		// Save image and object points in current CameraParams object
		cams.objectPoints = objPoints;
		cams.imagePoints1 = imgPoints1;
		cams.imagePoints2 = imgPoints2;

		// Extract camera parameters and retreive reprojection error on the calibration set of images
		double r_err = stereoCalibrate(	objPoints, imgPoints1, imgPoints2, cams.cameraMatrix1, cams.distortionCoeffs1, 
										cams.cameraMatrix2, cams.distortionCoeffs2, imageSize,
										cams.R, cams.T, cams.E, cams.F, CALIB_FIX_INTRINSIC);
		cams.reprojError = r_err;
		destroyAllWindows();
	}
}

// -------------------------------------------------------------------------------------//

void SaveParams(CameraParams &cam, boolean reproj)
{
	/*
	This function outputs the camera parameters of the CameraParams input object and additionally
	writes the parameters to an XML file found in the current directory called "cameraParams.xml".
	The boolean input is used to determine whether the CameraParams object had its parameters tested
	on a second set of calibration pattern images, in which case it should include the reprojection error
	of this test in the output and XML file.
	*/

	// Print out the relevant camera parameters and reprojection errors
	cout << "\n//---- " + cam.camName + " ----//" << endl;
	cout << "Camera Matrix:\n" << cam.cameraMatrix << endl;
	cout << "\nDistortion Coefficients:\n" << cam.distortionCoeffs << endl;
	cout << "\nReprojection Error (on calibration 1 images):\n" << cam.reprojError1 << endl;
	
	if (reproj) {
		cout << "\nReprojection error of Calib 1 params on Calib 2 images:\n" << cam.reprojError2 << endl;
	}

	// Write out the camera parameters and reprojection errors to an XML file
	FileStorage cameraParamsXML("cameraParams.xml", FileStorage::APPEND);
	cameraParamsXML << cam.camName + "_CameraMatrix" << cam.cameraMatrix;
	cameraParamsXML << cam.camName + "_DistCoeffs" << cam.distortionCoeffs;
	cameraParamsXML << cam.camName + "_ReprojError1" << cam.reprojError1; 

	if (reproj) {
		cameraParamsXML << cam.camName + "_ReprojError2" << cam.reprojError2;
	}
}

void SaveParams(StereoParams& cams)
{
	/*
	This function outputs the the camera parameters of each camera comprising a stereo-pair,
	their respective distortion coefficients, the rotation and translation matrices that describe
	the transformation from camera 2's co-ordinate system to camera 1's co-ordinate system, the
	essential matrix, the fundamental matrix, and the reprojection error for a stereo-pair of cameras.
	It then writes these parameters to a text file named "StereoParameters.txt".
	*/

	// Print out the relevant camera parameters and reprojection errors
	cout << "\n//---- " + cams.camNames + " ----//" << endl;
	cout << "Camera 1 Matrix:\n" << cams.cameraMatrix1 << endl;
	cout << "\nCamera 2 Matrix:\n" << cams.cameraMatrix2 << endl;
	cout << "\nCamera 1 Distortion Coefficients:\n" << cams.distortionCoeffs1 << endl;
	cout << "\nCamera 2 Distortion Coefficients:\n" << cams.distortionCoeffs2 << endl;
	cout << "\nRotation Matrix from Cam 1 to Cam 2:\n" << cams.R << endl;
	cout << "\nTranslation Matrix from Cam 1 to Cam 2:\n" << cams.T << endl;
	cout << "\nEssential Matrix:\n" << cams.E << endl;
	cout << "\nFundamental Matrix:\n" << cams.F << endl;
	cout << "\nReprojection Error:\n" << cams.reprojError << endl;

	// Write out the camera parameters and reprojection errors to an XML file
	FileStorage stereoParamsXML("stereoParams.xml", FileStorage::APPEND);
	stereoParamsXML << cams.camNames + "_CameraMatrix1" << cams.cameraMatrix1;
	stereoParamsXML << cams.camNames + "_CameraMatrix2" << cams.cameraMatrix2;
	stereoParamsXML << cams.camNames + "_DistCoeffs1" << cams.distortionCoeffs1;
	stereoParamsXML << cams.camNames + "_DistCoeffs2" << cams.distortionCoeffs2;
	stereoParamsXML << cams.camNames + "_RotMat" << cams.R;
	stereoParamsXML << cams.camNames + "_TransMat" << cams.T;
	stereoParamsXML << cams.camNames + "_EssentMat" << cams.E;
	stereoParamsXML << cams.camNames + "_FundMat" << cams.F;
	stereoParamsXML << cams.camNames + "_ReprojError" << cams.reprojError;
}

// -------------------------------------------------------------------------------------//

int main(int argc, char* argv[])
{
	// Command line argument options for calibration mode (argv[1]):
	//	0 = Only calibrate individual cameras
	//	1 = Only calibrate stereo-pairs
	//	2 = Calibrate both individual cameras and stereo-pairs
	// Set command line arguments under: Project Properties -> Debugging -> Command Line
	int calibMode = atoi(argv[1]);

	// Booleans to determine which calibration occurs (set in switch statement)
	boolean individual, stereo;

	// Checkerboard inner corners (width = 10 blocks, height = 6 blocks)
	int checkerboard[2]{ 9, 5 };

	// Switch statement to determine calibrations based on command line input
	switch (calibMode)
	{
		case 0:	// Individual Camera Calibration
			cout << "MODE: Individual Camera Calibrations Only:" << endl;
			individual = TRUE;
			stereo = FALSE;
			break;
		case 1:	// Stereo Camera Calibration
			cout << "MODE: Stereo-Pair Camera Calibrations Only:" << endl;
			individual = FALSE;
			stereo = TRUE;
			break;
		case 2:	// Both Individual and Stereo Calibrations
			cout << "MODE: Both Individual and Stereo-Pair Camera Calibrations:" << endl;
			individual = TRUE;
			stereo = TRUE;
			break;
		default:
			individual = FALSE;
			stereo = FALSE;
			cout << "ERROR: No calibration occurred. Check command line argument." << endl;
			break;
	}

	// Individual cameras calibration
	if (individual) 
	{
		cout << "\n//---------- INDIVIDUAL CAMERA CALIBRATIONS ----------//" << endl;

		// Paths to directories of individual calibration image sets for each camera
		map<string, string> calibImagePaths = { {"Cam1_Calib1","./Project_Pics/Calibration1/Cam1/*.bmp"},
												{"Cam2_Calib1","./Project_Pics/Calibration1/Cam2/*.bmp"},
												{"Cam3_Calib1","./Project_Pics/Calibration1/Cam3/*.bmp"},
												{"Cam1_Calib2","./Project_Pics/Calibration2/Cam1/*.bmp"},
												{"Cam2_Calib2","./Project_Pics/Calibration2/Cam2/*.bmp"},
												{"Cam3_Calib2","./Project_Pics/Calibration2/Cam3/*.bmp"}};

		//---------- FIRST CALIBRATION ----------//

		// Setting individual camera names
		cam1.camName = "CAMERA_1_CALIB_1";
		cam2.camName = "CAMERA_2_CALIB_1";
		cam3.camName = "CAMERA_3_CALIB_1";

		// Running the camera calibrations
		cout << "\nCamera 1 Individual Calibration\n" << endl;
		CalibrateCamera(checkerboard, cam1, calibImagePaths["Cam1_Calib1"]);
		cout << "\nCamera 2 Individual Calibration\n" << endl;
		CalibrateCamera(checkerboard, cam2, calibImagePaths["Cam2_Calib1"]);
		cout << "\nCamera 3 Individual Calibration\n" << endl;
		CalibrateCamera(checkerboard, cam3, calibImagePaths["Cam3_Calib1"]);

		//---------- SECOND CALIBRATION ----------//

		// Setting individual camera names
		cam1_calib2.camName = "CAMERA_1_CALIB_2";
		cam2_calib2.camName = "CAMERA_2_CALIB_2";
		cam3_calib2.camName = "CAMERA_3_CALIB_2";

		// Running the camera calibrations
		cout << "\nCamera 1 Second Individual Calibration\n" << endl;
		CalibrateCamera(checkerboard, cam1_calib2, calibImagePaths["Cam1_Calib2"]);
		cout << "\nCamera 2 Second Individual Calibration\n" << endl;
		CalibrateCamera(checkerboard, cam2_calib2, calibImagePaths["Cam2_Calib2"]);
		cout << "\nCamera 3 Second Individual Calibration\n" << endl;
		CalibrateCamera(checkerboard, cam3_calib2, calibImagePaths["Cam3_Calib2"]);

		// Calculating reprojection error of individual cameras on 2nd set of calibration images (after acquiring data)
		cam1.reprojError2 = ReprojectionErrors(cam1_calib2.objectPoints, cam1_calib2.imagePoints, cam1_calib2.rVecs,
							cam1_calib2.tVecs, cam1.cameraMatrix, cam1.distortionCoeffs);
		cam2.reprojError2 = ReprojectionErrors(cam2_calib2.objectPoints, cam2_calib2.imagePoints, cam2_calib2.rVecs,
							cam2_calib2.tVecs, cam2.cameraMatrix, cam2.distortionCoeffs);
		cam3.reprojError2 = ReprojectionErrors(cam3_calib2.objectPoints, cam3_calib2.imagePoints, cam3_calib2.rVecs,
							cam3_calib2.tVecs, cam3.cameraMatrix, cam3.distortionCoeffs);

		// Copy old camera parameters to backup XML file & save new parameters to XML file cameraParams.xml
		ifstream inFile("./cameraParams.xml");
		ofstream outFile("./cameraParamsCopy.xml");
		outFile << inFile.rdbuf();
		inFile.close();
		outFile.close();

		if (remove("cameraParams.xml") == 0) {
			cout << "\nSuccessfully deleted previous cameraParams.xml file. Writing current file..." << endl;
		}
		else {
			cout << "\nFailed to delete previous cameraParams.xml file." << endl;
		}

		// Saving all the camera parameters to the new XML file
		SaveParams(cam1, TRUE);
		SaveParams(cam2, TRUE);
		SaveParams(cam3, TRUE);
		SaveParams(cam1_calib2, FALSE);
		SaveParams(cam2_calib2, FALSE);
		SaveParams(cam3_calib2, FALSE);

		cout << "\nCompleted all individual camera calibrations.\n" << endl;
	}
	
	// Stereo-pair calibration
	if (stereo) 
	{
		cout << "\n//---------- STEREO-PAIR CALIBRATIONS ----------//" << endl;

		// Paths to directories of stereo calibration image sets for each camera pair
		map<string, string> stereoImagePaths = {{"Cam1_Cams21_Calib1","./Project_Pics/Calibration1/Cams21/Cam1/*.bmp"},
												{"Cam2_Cams21_Calib1","./Project_Pics/Calibration1/Cams21/Cam2/*.bmp"},
												{"Cam2_Cams32_Calib1","./Project_Pics/Calibration1/Cams32/Cam2/*.bmp"},
												{"Cam3_Cams32_Calib1","./Project_Pics/Calibration1/Cams32/Cam3/*.bmp"},
												{"Cam1_Cams21_Calib2","./Project_Pics/Calibration2/Cams21/Cam1/*.bmp"},
												{"Cam2_Cams21_Calib2","./Project_Pics/Calibration2/Cams21/Cam2/*.bmp"},
												{"Cam2_Cams32_Calib2","./Project_Pics/Calibration2/Cams32/Cam2/*.bmp"},
												{"Cam3_Cams32_Calib2","./Project_Pics/Calibration2/Cams32/Cam3/*.bmp"}};
		
		//---------- FIRST CALIBRATION ----------//
	
		// Setting stereo-pair names
		cams21.camNames = "CAMS21_CALIB1";
		cams32.camNames = "CAMS32_CALIB1";

		// Giving cameras intrinsics and distortion coeffs from their individual calibrations from XML file
		FileStorage camParams("cameraParams.xml", FileStorage::READ);

		// Camera Parameters 
		// Cams 21: Cam 2 left, Cam 1 right
		// Cams 32: Cam 3 left, Cam 2 right
		camParams["CAMERA_2_CALIB_1_CameraMatrix"] >> cams21.cameraMatrix1;
		camParams["CAMERA_1_CALIB_1_CameraMatrix"] >> cams21.cameraMatrix2;
		camParams["CAMERA_3_CALIB_1_CameraMatrix"] >> cams32.cameraMatrix1;
		camParams["CAMERA_2_CALIB_1_CameraMatrix"] >> cams32.cameraMatrix2;

		// Distortion Coefficients
		camParams["CAMERA_2_CALIB_1_DistCoeffs"] >> cams21.distortionCoeffs1;
		camParams["CAMERA_1_CALIB_1_DistCoeffs"] >> cams21.distortionCoeffs2;
		camParams["CAMERA_3_CALIB_1_DistCoeffs"] >> cams32.distortionCoeffs1;
		camParams["CAMERA_2_CALIB_1_DistCoeffs"] >> cams32.distortionCoeffs2;

		// Running the stereo calibrations
		cout << "\nStereo-Pair of Cameras 1 & 2 Calibration\n" << endl;
		StereoCalibrate(checkerboard, cams21, stereoImagePaths["Cam2_Cams21_Calib1"], stereoImagePaths["Cam1_Cams21_Calib1"]);
		cout << "\nStereo-Pair of Cameras 2 & 3 Calibration\n" << endl;
		StereoCalibrate(checkerboard, cams32, stereoImagePaths["Cam3_Cams32_Calib1"], stereoImagePaths["Cam2_Cams32_Calib1"]);
		
		//---------- SECOND CALIBRATION ----------//

		// Setting stereo-pair names
		cams21_calib2.camNames = "CAMS21_CALIB2";
		cams32_calib2.camNames = "CAMS32_CALIB2";

		// Giving cameras intrinsics and distortion coeffs from their individual calibrations 
	
		// Camera Parameters
		camParams["CAMERA_2_CALIB_2_CameraMatrix"] >> cams21_calib2.cameraMatrix1;
		camParams["CAMERA_1_CALIB_2_CameraMatrix"] >> cams21_calib2.cameraMatrix2;
		camParams["CAMERA_3_CALIB_2_CameraMatrix"] >> cams32_calib2.cameraMatrix1;
		camParams["CAMERA_2_CALIB_2_CameraMatrix"] >> cams32_calib2.cameraMatrix2;

		// Distortion Coefficients
		camParams["CAMERA_2_CALIB_2_DistCoeffs"] >> cams21_calib2.distortionCoeffs1;
		camParams["CAMERA_1_CALIB_2_DistCoeffs"] >> cams21_calib2.distortionCoeffs2;
		camParams["CAMERA_3_CALIB_2_DistCoeffs"] >> cams32_calib2.distortionCoeffs1;
		camParams["CAMERA_2_CALIB_2_DistCoeffs"] >> cams32_calib2.distortionCoeffs2;

		// Running the stereo calibrations
		cout << "\nStereo-Pair of Cameras 1 & 2 Second Calibration\n" << endl;
		StereoCalibrate(checkerboard, cams21_calib2, stereoImagePaths["Cam2_Cams21_Calib2"], stereoImagePaths["Cam1_Cams21_Calib2"]);
		cout << "\nStereo-Pair of Cameras 2 & 3 Second Calibration\n" << endl;
		StereoCalibrate(checkerboard, cams32_calib2, stereoImagePaths["Cam3_Cams32_Calib2"], stereoImagePaths["Cam2_Cams32_Calib2"]);
		
		// Copy old stereo-pair parameters to backup XML file & save new parameters to XML file cameraParams.xml
		ifstream inFile("./stereoParams.xml");
		ofstream outFile("./stereoParamsCopy.xml");
		outFile << inFile.rdbuf();
		inFile.close();
		outFile.close();

		if (remove("stereoParams.xml") == 0) {
			cout << "\nSuccessfully deleted previous stereoParams.xml file. Writing current file..." << endl;
		}
		else {
			cout << "\nFailed to delete previous stereoParams.xml file." << endl;
		}

		// Saving all the stereo-pair parameters to the new XML file
		SaveParams(cams21);
		SaveParams(cams32);
		SaveParams(cams21_calib2);
		SaveParams(cams32_calib2);

		cout << "\nCompleted all stereo-pair calibrations.\n" << endl;
	}

	return 0;
}