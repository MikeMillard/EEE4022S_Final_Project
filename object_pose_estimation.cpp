#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <chrono>

using namespace cv;
using namespace std;

// Size of every image (1000 x 800)
Size imageSize = Size(1000, 800);

// -------------------------------------------------------------------------------------//

Point getCentroid(InputArray Points)
{
    Point Coord;
    Moments mm = moments(Points, false);
    double moment10 = mm.m10;
    double moment01 = mm.m01;
    double moment00 = mm.m00;
    Coord.x = int(moment10 / moment00);
    Coord.y = int(moment01 / moment00);
    return Coord;
}

// -------------------------------------------------------------------------------------//

void undistortImages(map<string, string> imagesInPath, vector<Mat> camsIntrins, vector<Mat> camsDistCoeffs, map<string, string> imagesOutPath)
{
    /*
    This function takes the intrinsic parameters and distortion coefficients determined for each camera 
    during camera calibration and uses them to transform the images taken by each camera (for one instance 
    in time) such that they compensate for the lens distortion they contain (i.e. become undistorted images). 
    It then writes the images to a new folder containing each of the undistorted versions of the images.
    */
    
    // Reading in the images taken by each camera
    Mat cam1Image, cam2Image, cam3Image;
    cam1Image = imread(imagesInPath["Cam1"], IMREAD_UNCHANGED);
    cam2Image = imread(imagesInPath["Cam2"], IMREAD_UNCHANGED);
    cam3Image = imread(imagesInPath["Cam3"], IMREAD_UNCHANGED);

    // Reading in the intrinsics for each camera
    Mat cam1Intrins, cam2Intrins, cam3Intrins;
    cam1Intrins = camsIntrins[0];
    cam2Intrins = camsIntrins[1];
    cam3Intrins = camsIntrins[2];

    // Reading in the distortion coefficients for each camera
    Mat cam1DistCoeffs, cam2DistCoeffs, cam3DistCoeffs;
    cam1DistCoeffs = camsDistCoeffs[0];
    cam2DistCoeffs = camsDistCoeffs[1];
    cam3DistCoeffs = camsDistCoeffs[2];

    // Undistorting the images
    Mat cam1Undist, cam2Undist, cam3Undist;
    undistort(cam1Image, cam1Undist, cam1Intrins, cam1DistCoeffs);
    undistort(cam2Image, cam2Undist, cam2Intrins, cam2DistCoeffs);
    undistort(cam3Image, cam3Undist, cam3Intrins, cam3DistCoeffs);

    // Saving the images to the specified directories
    imwrite(imagesOutPath["Cam1"], cam1Undist);
    imwrite(imagesOutPath["Cam2"], cam2Undist);
    imwrite(imagesOutPath["Cam3"], cam3Undist);
}

// -------------------------------------------------------------------------------------//

vector<float> linspace(float start_in, float end_in, int num_in)
{
    vector<float> linspaced;

    float start = static_cast<float>(start_in);
    float end = static_cast<float>(end_in);
    float num = static_cast<float>(num_in);

    if (num == 0) { return linspaced; }
    if (num == 1)
    {
        linspaced.push_back(start);
        return linspaced;
    }

    float delta = (end - start) / (num - 1);
  
    for (int i = 0; i < num - 1; i++)
    {
        linspaced.push_back(start + delta * i);
    }
    linspaced.push_back(end);

    return linspaced;
}

// -------------------------------------------------------------------------------------//

Mat inhomogToHomog(Point3f& point3D) {
    Mat homog3DMat = (Mat_<float>(4, 1) << point3D.x, point3D.y, point3D.z, 1);
    return homog3DMat;
}

// -------------------------------------------------------------------------------------//

Point2f homogToInhomog2D(Mat& homog2DMat) {
    Point2f point2D(homog2DMat.at<float>(0, 0) / homog2DMat.at<float>(2, 0),
                    homog2DMat.at<float>(1, 0) / homog2DMat.at<float>(2, 0));
    return point2D;
}

// -------------------------------------------------------------------------------------//

Point2f matToPoint2D(Mat& mat2DPoint) {
    Point2f point2D;
    point2D.x = mat2DPoint.at<float>(0, 0);
    point2D.y = mat2DPoint.at<float>(1, 0);
    return point2D;
}

// -------------------------------------------------------------------------------------//

float euclideanDist(Point2f& ptA, Point2f& ptB)
{
    Point2f diff = ptA - ptB;
    return sqrt(diff.x * diff.x + diff.y * diff.y);
}

// -------------------------------------------------------------------------------------//

bool compareFloat(float x, float y, float epsilon = 0.00001f) {
    // Compare two floating point variables up to 5 decimal places
    if (fabs(x - y) < epsilon)
        return true; 
    return false; 
}

// -------------------------------------------------------------------------------------//

void printVecElems(vector<float>& vect) {
    cout << endl;
    for (int i = 0; i < vect.size(); i++) {
        cout << vect[i] << endl;
    }
    cout << endl;
}

// -------------------------------------------------------------------------------------//

void printVecElems(vector<Point3f>& vect) {
    cout << endl;
    for (int i = 0; i < vect.size(); i++) {
        cout << "{" << vect[i].x << ", " << vect[i].y << ", " << vect[i].z << "}" << endl;
    }
    cout << endl;
}

// -------------------------------------------------------------------------------------//

void printVecElems(vector<Point2f>& vect) {
    cout << endl;
    for (int i = 0; i < vect.size(); i++) {
        cout << "{" << vect[i].x << ", " << vect[i].y << "}" << endl;
    }
    cout << endl;
}

// -------------------------------------------------------------------------------------//

Mat rotation(float thetaX, float thetaY, float thetaZ) {
    // Rotation matrices about x, y, and z axes are all 3x3 square matrices of 32-bit floats
    Mat rotX = (Mat_<float>(3, 3) <<    1, 0, 0,
                                        0, cosf(thetaX), -sinf(thetaX),
                                        0, sinf(thetaX), cosf(thetaX));

    Mat rotY = (Mat_<float>(3, 3) <<    cosf(thetaY), 0, sinf(thetaY),
                                        0, 1, 0,
                                        -sinf(thetaY), 0, cosf(thetaY));

    Mat rotZ = (Mat_<float>(3, 3) <<    cosf(thetaZ), -sinf(thetaZ), 0,
                                        sinf(thetaZ), cosf(thetaZ), 0,
                                        0, 0, 1);

    // Overall rotation matrix (multiplication order: x x y x z)
    Mat rot = rotZ * rotY * rotX;
    return rot;
}

// -------------------------------------------------------------------------------------//

Mat translation(float tX, float tY, float tZ) {
    Mat trans = (Mat_<float>(3, 1) << tX, tY, tZ);
    return trans;
}

// -------------------------------------------------------------------------------------//

Mat getPose6DoF(vector<float> poseParams) {
    // Pose parameters must be given in the following sequence:
    //      { thetaX, thetaY, thetaZ, tX, tY, tZ }
    Mat pose6DoF = Mat(3, 4, CV_32F);
    Mat rot = rotation(poseParams[0], poseParams[1], poseParams[2]);
    Mat trans = translation(poseParams[3], poseParams[4], poseParams[5]);
    hconcat(rot, trans, pose6DoF);

    return pose6DoF;
}

// -------------------------------------------------------------------------------------//

vector<Mat> getHomogeneousPoints(vector<Point3f> inhomog3DCylPoints) {
    vector<Mat> homog3DCylPoints;
    for (int i = 0; i < inhomog3DCylPoints.size(); i++) {
        Mat homog3DCylPoint = inhomogToHomog(inhomog3DCylPoints[i]);
        homog3DCylPoints.push_back(homog3DCylPoint);
    }

    return homog3DCylPoints;
}

// -------------------------------------------------------------------------------------//

Mat sharpenImage(Mat camImage, bool display) {
    
    Mat camImageCopy = camImage.clone();
    Mat lapKernel = (Mat_<float>(3, 3) <<   1, 1, 1,
                                            1, -8, 1,
                                            1, 1, 1);
    Mat lapImage;
    filter2D(camImageCopy, lapImage, CV_32F, lapKernel);
    camImageCopy.convertTo(camImageCopy, CV_32F);
    Mat sharpImage = camImageCopy - lapImage;
    sharpImage.convertTo(sharpImage, CV_8UC3);
    Mat sharpImageGrey;
    cvtColor(sharpImage, sharpImageGrey, COLOR_BGR2GRAY);

    // Display the sharpened image
    if (display) {
        imshow("Sharpened grey image", sharpImageGrey);
        waitKey();
    }

    return sharpImageGrey;
}

// -------------------------------------------------------------------------------------//

Mat binarizeImage(Mat sharpGreyImage, bool display) {
    
    Mat sharpGreyImageCopy = sharpGreyImage.clone(), imageBlur, 
        imageBlurBinary, imageSharpBinary, finalImageBinary;


    // Slight Gaussian blur 3x3 window
    GaussianBlur(sharpGreyImageCopy, imageBlur, Size(3, 3), 1);

    // Binarizing blurry image using relatively low binary threshold
    threshold(imageBlur, imageBlurBinary, 50, 255, THRESH_BINARY);

    // Binarizing sharp image using higher binary threshold or Otsu threshold
    threshold(sharpGreyImageCopy, imageSharpBinary, 50, 255, THRESH_OTSU);
    
    // Modifying top sixth of the final binarized image due to poor segmentation
    Mat topSixthSharp = imageSharpBinary.rowRange(0, imageSize.height/6);
    vconcat(topSixthSharp, imageBlurBinary.rowRange(imageSize.height/6, imageSize.height), finalImageBinary);
    
    // Display the binarized image
    if (display) {
        imshow("Final binary image", finalImageBinary);
        waitKey();
    }

    return finalImageBinary;
}

// -------------------------------------------------------------------------------------//

void testAllBinarizations(Mat camGrey, Mat camGreySharp, bool display) {
    
    // DEMONSTRATING ALL BINARIZATION OPERATIONS ON GIVEN CAMERA IMAGE

    // 6 different binary images considered:
    //      1. Binarize greyscale image      
    //      2. Binarize blurred greyscale image
    //      3. Binarize sharp greyscale image
    //      4. Binarize sharp blurred greyscale image
    //      5. Binarize close normalized greyscale image
    //      6. Binarize close normalized blurred greyscale image

    Mat camGreyClose, camGreyCloseNorm = camGrey.clone();

    // Perform closing operation (dilate image first, then erode)
    Mat kernel(7, 7, CV_8U, Scalar(1));
    morphologyEx(camGreySharp, camGreyClose, MORPH_CLOSE, kernel);
    
    // Normalizing original greyscale image using close images
    for (int u = 0; u < camGrey.rows; u++) {
        for (int v = 0; v < camGrey.cols; v++) {

            // Normalize sharp greyscale using close
            float normCloseRatioAtPixel = ((float)camGreySharp.at<uchar>(u, v) / (float)camGreyClose.at<uchar>(u, v));
            camGreyCloseNorm.at<uchar>(u, v) = ceil((float)camGrey.at<uchar>(u, v) * normCloseRatioAtPixel);

        }
    }

    // Gaussian blurring
    Mat camGreyBlur, camGreySharpBlur, camGreyCloseNormBlur;
    GaussianBlur(camGrey, camGreyBlur, Size(3, 3), 1);
    GaussianBlur(camGreySharp, camGreySharpBlur, Size(3, 3), 1);
    GaussianBlur(camGreyCloseNorm, camGreyCloseNormBlur, Size(3, 3), 1);

    // Binarizing all of the different options
    Mat camGreyBinary, camGreyBlurBinary,
        camGreySharpBinary, camGreySharpBlurBinary,
        camGreyCloseNormBinary, camGreyCloseNormBlurBinary;

    // Cam greyscale and blur
    threshold(camGrey, camGreyBinary, 50, 255, THRESH_BINARY);
    threshold(camGreyBlur, camGreyBlurBinary, 50, 255, THRESH_BINARY);
    
    // Cam sharp greyscale and blur
    threshold(camGreySharp, camGreySharpBinary, 50, 255, THRESH_BINARY);
    threshold(camGreySharpBlur, camGreySharpBlurBinary, 50, 255, THRESH_BINARY);
    
    // Cam close normalized greyscale and blur
    threshold(camGreyCloseNorm, camGreyCloseNormBinary, 50, 255, THRESH_BINARY);
    threshold(camGreyCloseNormBlur, camGreyCloseNormBlurBinary, 50, 255, THRESH_BINARY); 

    // Display all of the binarized images
    if (display) {
        // Cam greyscale and blur
        imshow("Cam grey binarized", camGreyBinary);
        imshow("Cam grey blur binarized", camGreyBlurBinary);
        // Cam sharp greyscale and blur
        imshow("Cam grey sharp binarized", camGreySharpBinary);
        imshow("Cam grey sharp blur binarized", camGreySharpBlurBinary);
        // Cam close normalized greyscale and blur
        imshow("Cam grey close normalized binarized", camGreyCloseNormBinary);
        imshow("Cam grey close normalized blur binarized", camGreyCloseNormBlurBinary);

        waitKey();
    }
    
}

// -------------------------------------------------------------------------------------//


int main(int argc, char** argv)
{
    // Maps to input distorted object images (fill in image number)
    map<string, string> babyCreamPathIn = { {"Cam1","./Project_Pics/JJ_Baby_Cream/Original/Cam1/baby_cream_1_cam1.bmp"},
                                            {"Cam2","./Project_Pics/JJ_Baby_Cream/Original/Cam2/baby_cream_1_cam2.bmp"},
                                            {"Cam3","./Project_Pics/JJ_Baby_Cream/Original/Cam3/baby_cream_1_cam3.bmp"}};

    map<string, string> noLyeOrangePathIn = { {"Cam1","./Project_Pics/no_lye_orange/Original/Cam1/no_lye_orange_1_cam1.bmp"},
                                              {"Cam2","./Project_Pics/no_lye_orange/Original/Cam2/no_lye_orange_1_cam2.bmp"},
                                              {"Cam3","./Project_Pics/no_lye_orange/Original/Cam3/no_lye_orange_1_cam3.bmp"}};

    map<string, string> noLyePinkPathIn = { {"Cam1","./Project_Pics/no_lye_pink/Original/Cam1/no_lye_pink_1_cam1.bmp"},
                                            {"Cam2","./Project_Pics/no_lye_pink/Original/Cam2/no_lye_pink_1_cam2.bmp"},
                                            {"Cam3","./Project_Pics/no_lye_pink/Original/Cam3/no_lye_pink_1_cam3.bmp"}};

    map<string, string> storkPathIn = { {"Cam1","./Project_Pics/stork/Original/Cam1/stork_1_cam1.bmp"},
                                        {"Cam2","./Project_Pics/stork/Original/Cam2/stork_1_cam2.bmp"},
                                        {"Cam3","./Project_Pics/stork/Original/Cam3/stork_1_cam3.bmp"}};

    // Maps to output undistorted object images (fill in image number)
    map<string, string> babyCreamPathOut = {{"Cam1","./Project_Pics/JJ_Baby_Cream/Undistorted/Cam1/baby_cream_1_cam1.bmp"},
                                            {"Cam2","./Project_Pics/JJ_Baby_Cream/Undistorted/Cam2/baby_cream_1_cam2.bmp"},
                                            {"Cam3","./Project_Pics/JJ_Baby_Cream/Undistorted/Cam3/baby_cream_1_cam3.bmp"}};

    map<string, string> noLyeOrangePathOut = {  {"Cam1","./Project_Pics/no_lye_orange/Undistorted/Cam1/no_lye_orange_1_cam1.bmp"},
                                                {"Cam2","./Project_Pics/no_lye_orange/Undistorted/Cam2/no_lye_orange_1_cam2.bmp"},
                                                {"Cam3","./Project_Pics/no_lye_orange/Undistorted/Cam3/no_lye_orange_1_cam3.bmp"}};

    map<string, string> noLyePinkPathOut = {{"Cam1","./Project_Pics/no_lye_pink/Undistorted/Cam1/no_lye_pink_1_cam1.bmp"},
                                            {"Cam2","./Project_Pics/no_lye_pink/Undistorted/Cam2/no_lye_pink_1_cam2.bmp"},
                                            {"Cam3","./Project_Pics/no_lye_pink/Undistorted/Cam3/no_lye_pink_1_cam3.bmp"}};

    map<string, string> storkPathOut = {{"Cam1","./Project_Pics/stork/Undistorted/Cam1/stork_1_cam1.bmp"},
                                        {"Cam2","./Project_Pics/stork/Undistorted/Cam2/stork_1_cam2.bmp"},
                                        {"Cam3","./Project_Pics/stork/Undistorted/Cam3/stork_1_cam3.bmp"}};

    // Maps to binarized object images (fill in image number)
    map<string, string> babyCreamPathBin = { {"Cam1","./Project_Pics/JJ_Baby_Cream/Binarized/Cam1/baby_cream_1_cam1.bmp"},
                                            {"Cam2","./Project_Pics/JJ_Baby_Cream/Binarized/Cam2/baby_cream_1_cam2.bmp"},
                                            {"Cam3","./Project_Pics/JJ_Baby_Cream/Binarized/Cam3/baby_cream_1_cam3.bmp"} };

    map<string, string> noLyeOrangePathBin = { {"Cam1","./Project_Pics/no_lye_orange/Binarized/Cam1/no_lye_orange_1_cam1.bmp"},
                                                {"Cam2","./Project_Pics/no_lye_orange/Binarized/Cam2/no_lye_orange_1_cam2.bmp"},
                                                {"Cam3","./Project_Pics/no_lye_orange/Binarized/Cam3/no_lye_orange_1_cam3.bmp"} };

    map<string, string> noLyePinkPathBin = { {"Cam1","./Project_Pics/no_lye_pink/Binarized/Cam1/no_lye_pink_1_cam1.bmp"},
                                            {"Cam2","./Project_Pics/no_lye_pink/Binarized/Cam2/no_lye_pink_1_cam2.bmp"},
                                            {"Cam3","./Project_Pics/no_lye_pink/Binarized/Cam3/no_lye_pink_1_cam3.bmp"} };

    map<string, string> storkPathBin = { {"Cam1","./Project_Pics/stork/Binarized/Cam1/stork_1_cam1.bmp"},
                                        {"Cam2","./Project_Pics/stork/Binarized/Cam2/stork_1_cam2.bmp"},
                                        {"Cam3","./Project_Pics/stork/Binarized/Cam3/stork_1_cam3.bmp"} };

    // Create undistorted version of each image

    // Intrinsics of each camera (3x3 matrices)
    Mat cam1Intrins, cam2Intrins, cam3Intrins;
    vector<Mat> camsIntrins;

    FileStorage camParams("cameraParams.xml", FileStorage::READ);
    camParams["CAMERA_1_CALIB_1_CameraMatrix"] >> cam1Intrins;  cam1Intrins.convertTo(cam1Intrins, CV_32F);
    camParams["CAMERA_2_CALIB_1_CameraMatrix"] >> cam2Intrins;  cam2Intrins.convertTo(cam2Intrins, CV_32F);
    camParams["CAMERA_3_CALIB_1_CameraMatrix"] >> cam3Intrins;  cam3Intrins.convertTo(cam3Intrins, CV_32F);
    camsIntrins.push_back(cam1Intrins); camsIntrins.push_back(cam2Intrins); camsIntrins.push_back(cam3Intrins);

    // Distortion coefficients of each camera (1x5 matrices)
    Mat cam1DistCoeffs, cam2DistCoeffs, cam3DistCoeffs;
    vector<Mat> camsDistCoeffs;

    // Read in distortion coeffiecients for each camera from XML file
    camParams["CAMERA_1_CALIB_1_DistCoeffs"] >> cam1DistCoeffs; cam1DistCoeffs.convertTo(cam1DistCoeffs, CV_32F);
    camParams["CAMERA_2_CALIB_1_DistCoeffs"] >> cam2DistCoeffs; cam2DistCoeffs.convertTo(cam2DistCoeffs, CV_32F);
    camParams["CAMERA_3_CALIB_1_DistCoeffs"] >> cam3DistCoeffs; cam3DistCoeffs.convertTo(cam3DistCoeffs, CV_32F);
    camsDistCoeffs.push_back(cam1DistCoeffs); camsDistCoeffs.push_back(cam2DistCoeffs); camsDistCoeffs.push_back(cam3DistCoeffs);

    // Undistort each of the images
    
    // J&J Baby Cream undistorted images
    // Undistort timer start
    auto startUndist = chrono::high_resolution_clock::now();
    undistortImages(babyCreamPathIn, camsIntrins, camsDistCoeffs, babyCreamPathOut);
    // Undistort timer stop
    auto stopUndist = chrono::high_resolution_clock::now();
    auto durationUndist = chrono::duration_cast<chrono::milliseconds>(stopUndist - startUndist);
    // Convert to seconds
    float durationUndistSecs = (float)durationUndist.count() / 1000;
    cout << "\nTime to undistort all 3 images: " << durationUndistSecs << " seconds" << endl;

    // No-Lye Orange undistorted images
    //undistortImages(noLyeOrangePathIn, camsIntrins, camsDistCoeffs, noLyeOrangePathOut);

    // No-Lye Pink undistorted images
    //undistortImages(noLyePinkPathIn, camsIntrins, camsDistCoeffs, noLyePinkPathOut);

    // Stork undistorted images
    //undistortImages(storkPathIn, camsIntrins, camsDistCoeffs, storkPathOut);

    // Working with JJ Baby Cream Stereo Pairs
    
    // Undistorted colour images
    Mat cam1Image, cam2Image, cam3Image;
    cam1Image = imread(babyCreamPathOut["Cam1"], IMREAD_UNCHANGED);
    cam2Image = imread(babyCreamPathOut["Cam2"], IMREAD_UNCHANGED);
    cam3Image = imread(babyCreamPathOut["Cam3"], IMREAD_UNCHANGED);
    
    // Greyscale equivalents
    Mat cam1Grey, cam2Grey, cam3Grey;
    cvtColor(cam1Image, cam1Grey, COLOR_BGR2GRAY); 
    cvtColor(cam2Image, cam2Grey, COLOR_BGR2GRAY); 
    cvtColor(cam3Image, cam3Grey, COLOR_BGR2GRAY);

    // -------------------------------------------------------------------------------------//
    
    // Stereo Cameras
    FileStorage stereoParams("stereoParams.xml", FileStorage::READ);

    // Cam2 to cam1
    Mat rotC2toC1 = Mat(3, 3, CV_32F), transC2toC1 = Mat(3, 1, CV_32F);
    stereoParams["CAMS21_CALIB1_RotMat"] >> rotC2toC1;      rotC2toC1.convertTo(rotC2toC1, CV_32F);
    stereoParams["CAMS21_CALIB1_TransMat"] >> transC2toC1;  transC2toC1.convertTo(transC2toC1, CV_32F);
    transC2toC1 = 10 * transC2toC1; // Conversion from cm to mm

    // Cam3 to cam2
    Mat rotC3toC2 = Mat(3, 3, CV_32F), transC3toC2 = Mat(3, 1, CV_32F);
    stereoParams["CAMS32_CALIB1_RotMat"] >> rotC3toC2;      rotC3toC2.convertTo(rotC3toC2, CV_32F);
    stereoParams["CAMS32_CALIB1_TransMat"] >> transC3toC2;  transC3toC2.convertTo(transC3toC2, CV_32F);
    transC3toC2 = 10 * transC3toC2; // Conversion from cm to mm


    // CAMERA SPECS (VEN-161-61U3C USB3.0)
    // Image size: 1/2.9" (Sensor size)
    // Max res: 1440x1080 (Actual taken was 1000x800 with offset of (x_off, y_off) = (220, 140)
    // Dimensions: 35x35x8.1 mm
    // Pixel size: 3.45 um = 0.00345 mm
    // Frame rate: 61.3 Hz
    // Pixel bit depth: 8-bit/10-bit

    // LENS SPECS (CCTV IR CS Mount Lens)
    // Fixed focal length: 6mm
    // F-stop: F1.2
    // Image size: 1/3" format
    // Angle of View: 50 degrees
    
    // Sharpen all 3 images using Laplacian filter
    bool dispSharp = false;
    // Timer start
    auto startSharp = chrono::high_resolution_clock::now();
    Mat cam1GreySharp = sharpenImage(cam1Image, dispSharp);
    Mat cam2GreySharp = sharpenImage(cam2Image, dispSharp);
    Mat cam3GreySharp = sharpenImage(cam3Image, dispSharp);
    // Timer stop
    auto stopSharp = chrono::high_resolution_clock::now();
    auto durationSharp = chrono::duration_cast<chrono::milliseconds>(stopSharp - startSharp);
    // Convert to seconds
    float durationSharpSecs = (float)durationSharp.count() / 1000;
    cout << "\nTime to sharpen all 3 images: " << durationSharpSecs << " seconds" << endl;

    // Binarize all 3 images
    bool dispBin = false;
    // Binarize timer start
    auto startBin = chrono::high_resolution_clock::now();
    Mat cam1Binarized = binarizeImage(cam1GreySharp, dispBin);
    Mat cam2Binarized = binarizeImage(cam2GreySharp, dispBin);
    Mat cam3Binarized = binarizeImage(cam3GreySharp, dispBin);
    // Binarize timer stop
    auto stopBin = chrono::high_resolution_clock::now();
    auto durationBin = chrono::duration_cast<chrono::milliseconds>(stopBin - startBin);
    // Convert to seconds
    float durationBinSecs = (float)durationBin.count() / 1000;
    cout << "\nTime to binarize all 3 images: " << durationBinSecs << " seconds" << endl;

    // Save the binarized images for each camera
    imwrite(babyCreamPathBin["Cam1"], cam1Binarized);
    imwrite(babyCreamPathBin["Cam2"], cam2Binarized);
    imwrite(babyCreamPathBin["Cam3"], cam3Binarized);

    // Show all binarizations considered for given camera image
    bool dispTests = false;
    testAllBinarizations(cam3Grey, cam3GreySharp, dispTests);
   
    // Contours in cam 3
    // Init pose timer start
    auto startPose = chrono::high_resolution_clock::now();
    // Find contours
    vector<vector<Point>> contoursCam3;
    vector<Vec4i> hierarchyCam3;
    Mat drawingCam3 = cam3Image.clone();
    findContours(cam3Binarized, contoursCam3, hierarchyCam3, RETR_TREE, CHAIN_APPROX_SIMPLE);
    int maxContourSize = 0;             // no. of points in max contour
    int maxContourIndex = 0;            // index of max contour in contours vector
    vector<Point> maxContourCam3;       // vector of points for largest contour
    vector<vector<Point>> hull(contoursCam3.size());
    for (size_t i = 0; i < contoursCam3.size(); i++)
    {
        int contourSize = contoursCam3[i].size();
        if (contourSize > maxContourSize) {
            maxContourSize = contourSize;
            maxContourIndex = i;
        }
        convexHull(contoursCam3[i], hull[i]);
    }

    vector<Point> maxContourPolyCam3;
    approxPolyDP(contoursCam3[maxContourIndex], maxContourPolyCam3, 3, true);
    // Bounding rect
    Rect boundRectCam3 = boundingRect(maxContourPolyCam3);
    // Bounding rect centre
    Point2f rectCentreCam3((boundRectCam3.br().x + boundRectCam3.tl().x) / 2.0, (boundRectCam3.br().y + boundRectCam3.tl().y) / 2.0);
    // Principle point
    Point2f principle(cam3Intrins.at<float>(0, 2), cam3Intrins.at<float>(1, 2));
    // Initial translation guesses for object pose 
    float zInitial = 290;              // Measured in mm
    Mat homogRectCentreCam3 = zInitial*(Mat_<float>(3, 1) <<    rectCentreCam3.x,
                                                                rectCentreCam3.y, 
                                                                1);
    Mat rectCentre3DCam3 = cam3Intrins.inv() * homogRectCentreCam3;
    float xInitial = rectCentre3DCam3.at<float>(0, 0);   // For x right is postive 
    float yInitial = rectCentre3DCam3.at<float>(1, 0);   // For y down is positive  
    // Init pose timer stop
    auto stopPose = chrono::high_resolution_clock::now();
    auto durationPose = chrono::duration_cast<chrono::milliseconds>(stopPose - startPose);
    // Convert to seconds
    float durationPoseSecs = (float)durationPose.count() / 1000;
    cout << "\nTime to generate initial cam 3 pose: " << durationPoseSecs << " seconds" << endl;
    // Total image processing time
    float totalDuration = durationUndistSecs + durationSharpSecs + durationBinSecs + durationPoseSecs;
    cout << "\n Total time taken: " << totalDuration << " seconds\n" << endl;

    // Draw all above operations (rect, convex hull, etc) on image
    // Bounding rect
    rectangle(drawingCam3, boundRectCam3.tl(), boundRectCam3.br(), Scalar(0, 0, 255), 2);
    putText(drawingCam3, "Bounding rect", Point(boundRectCam3.tl().x, boundRectCam3.br().y) + Point(10, 20), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 0, 255), 1, 8);
    // Bounding rect centre
    circle(drawingCam3, rectCentreCam3, 1, Scalar(0, 0, 255), 4, 8);
    putText(drawingCam3, "Rect centre", rectCentreCam3 + Point2f(15, -5), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(255, 255, 0), 1, 8);
    // Principle point
    circle(drawingCam3, principle, 1, Scalar(0, 0, 255), 4, 8);
    putText(drawingCam3, "Principle point", principle + Point2f(15, 10), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(255, 255, 0), 1, 8);
    // Arrowed line from principle point to rect centre
    arrowedLine(drawingCam3, principle + Point2f(-6, -2), rectCentreCam3 + Point2f(5, 3), Scalar(0, 255, 0), 2, LINE_AA, 0.9);
    // Convex hull (max contour)
    drawContours(drawingCam3, hull, maxContourIndex, Scalar(0, 255, 0), 2, 8);
    putText(drawingCam3, "Convex hull", Point(boundRectCam3.tl().x, boundRectCam3.br().y) + Point(30, -40), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 255, 0), 1, 8);
    // Display the image
    imshow("Contours and Rect Centre", drawingCam3);

    // ------------ CYLINDER MODEL --------------- //
    // Cylinder parameters (Baby cream: height: 84mm, print radius: 45.5mm, top radius: 43.5mm) 
    // (z-axis vertical, x-axis horizontal, y-axis into screen)

    // Approximate Mesh
    vector<float> zv;       // z values (vertical)
    vector<float> thvd;     // theta values (around z-axis)
    vector<float> rc;       // radius values
    vector<float> temp;     // temp vector for concatenating vectors for different object heights

    // One disk per mm
    bool inclBottom = false;
    int numAngles = 360;
    float bottom = 8, middle = 57, top = 19, level = 0; // mm measurements
    float rBot, rMid = 45.5, rTop = 43.5;
    
    thvd = linspace(0.0, 360.0, numAngles);
    
    if (inclBottom) {
        // Bottom 8mm (0 - 8mm) radius 43.5:
        zv = linspace(level, bottom - 1, bottom);
        rc = linspace(rMid, rTop, bottom);
        level += bottom;
    }

    // Print 57mm (8 - 65mm) radius 45.5:
    temp = linspace(level, level + middle, middle + 1);
    zv.insert(end(zv), begin(temp), end(temp));
    temp = linspace(rMid, rMid, middle + 1);
    rc.insert(end(rc), begin(temp), end(temp));
    level += middle;

    // Top 19mm (65 - 84mm) radius 43.5:
    temp = linspace(level + 1, level + top, top);
    zv.insert(end(zv), begin(temp), end(temp));
    temp = linspace(rTop, rTop, top);
    rc.insert(end(rc), begin(temp), end(temp));
    level += top;

    // Creating vector of 3D cylinder points
    Point3f cylPoint;
    vector<Point3f> cyl3DPoints; 

    for (int i = 0; i < (int)zv.size(); i++) {
        for (int j = 0; j < (int)thvd.size(); j++) {

            // 3D co-ords of each cylinder point
            cylPoint.x = rc[i] * cos(thvd[j]* CV_PI / 180);
            cylPoint.y = rc[i] * sin(thvd[j] * CV_PI / 180);
            cylPoint.z = zv[i];
            cyl3DPoints.push_back(cylPoint);

        }
    }

    // Middle of cylinder at the bottom is the origin of the world co-ordinate system
    
    // Boolean to display image of projected points
    bool disp = true;

    // Rotation angles about each axis (from camera 3 to world)
    // Clockwise = +ve, Anti-clockwise = -ve
    float thetaX = 90 * CV_PI / 180;
    float thetaY = 0 * CV_PI / 180;
    float thetaZ = 0 * CV_PI / 180;

    // Translation in each axis direction (from camera 3 to world) 
    // Calculated above using centre of bounding rect
    float tX = xInitial;
    float tY = yInitial + level/2.0;
    float tZ = zInitial;

    // Relative to Cam 3
    
    // Object pose in camera 3 space
    vector<float> objectPoseParams = { thetaX, thetaY, thetaZ, tX, tY, tZ };

    // Camera 3 extrins (world to cam 3)
    Mat objectPoseCam3 = getPose6DoF(objectPoseParams);
    
    // Homogeneous camera 3 extrinsics
    Mat homogObjectPoseCam3, bottomRow = (Mat_<float>(1, 4) << 0, 0, 0, 1);
    vconcat(objectPoseCam3, bottomRow, homogObjectPoseCam3);

    // Camera 3 matrix
    /*
        P3 = K3[R3|t3]
    */
    Mat P3 = cam3Intrins * objectPoseCam3;

    
    // Extrins from C3 to C2 co-ordinates
    Mat RtC3toC2;
    hconcat(rotC3toC2, transC3toC2, RtC3toC2);

    // Camera 2 extrins (world to cam 2)
    Mat objectPoseCam2 = RtC3toC2 * homogObjectPoseCam3;

    // Homogeneous camera 2 extrinsics
    Mat homogObjectPoseCam2;
    vconcat(objectPoseCam2, bottomRow, homogObjectPoseCam2);

    // Camera 2 matrix
    /*
        P2 = K2[R2|t2]
        where [R2|t2] = [R3to2|t3to2]*homogObjectPoseCam3
    */
    Mat P2 = cam2Intrins * objectPoseCam2;

    // Extrins from C2 to C1 co-ordinates
    Mat RtC2toC1;
    hconcat(rotC2toC1, transC2toC1, RtC2toC1);

    // Camera 1 extrins (world to cam 1)
    Mat objectPoseCam1 = RtC2toC1 * homogObjectPoseCam2;

    // Camera 1 matrix
    /*
        P1 = K1[R1|t1]
        where [R1|t1] = [R2to1|t2to1]*homogObjectPoseCam2
    */
    Mat P1 = cam1Intrins * objectPoseCam1;
  
    // Steps: Starting with camera 3 projection
    
    // 1. homogeneous cylinder coords in world space [Xw Yw Zw 1]^T
    vector<Mat> homog3DCylPoints = getHomogeneousPoints(cyl3DPoints);    
    // 2. multiply with cam 3 extrins to get cylinder points in cam 3 space [Xc3 Yc3 Zc3]^T
    vector<Mat> cylPointsCam3;
    for (int i = 0; i < homog3DCylPoints.size(); i++) {
        Mat cylPointCam3 = objectPoseCam3 * homog3DCylPoints[i];
        cylPointsCam3.push_back(cylPointCam3);
    }
    // 3. multiply with cam 3 intrins to get homogeneous pixel points [z3x3 z3y3 z3]^T
    vector<Mat> homogPixPointsCam3;
    for (int k = 0; k < cylPointsCam3.size(); k++) {
        Mat homogPixPointCam3 = cam3Intrins * cylPointsCam3[k];
        homogPixPointsCam3.push_back(homogPixPointCam3);
    }
    // 4. convert to inhomogeneous pixel points [x3 y3]^T
    vector<Point2f> pixPointsCam3;
    for (int k = 0; k < homogPixPointsCam3.size(); k++) {
        Point2f pixPointCam3 = homogToInhomog2D(homogPixPointsCam3[k]);
        pixPointsCam3.push_back(pixPointCam3);
    }
    // 5. draw cylinder points onto image
    cvtColor(cam3Binarized, cam3Binarized, COLOR_GRAY2BGR);
    Mat cam3BinarizedCopy = cam3Binarized.clone();
    for (int l = 0; l < pixPointsCam3.size(); l++) {
        circle(cam3BinarizedCopy, pixPointsCam3[l], 1, Scalar(0, 0, 255), 1, 8);
    }
    // 6. Display image
    imshow("Cam 3 Projection", cam3BinarizedCopy);

    // Now projecting into camera 2 frame
    // 7. multiple world cylinder points with cam 2 extrins to get points in cam 2 space [Xc2, Yc2, Zc2]^T
    vector<Mat> cylPointsCam2;
    for (int i = 0; i < homog3DCylPoints.size(); i++) {
        Mat cylPointCam2 = objectPoseCam2 * homog3DCylPoints[i];
        cylPointsCam2.push_back(cylPointCam2);
    }
    // 8. multiply with cam 2 intrins to get homogeneous pixel points [z2x2 z2y2 z2]^T
    vector<Mat> homogPixPointsCam2;
    for (int k = 0; k < cylPointsCam2.size(); k++) {
        Mat homogPixPointCam2 = cam2Intrins * cylPointsCam2[k];
        homogPixPointsCam2.push_back(homogPixPointCam2);
    }
    // 9. convert to inhomogeneous pixel points [x2 y2]^T
    vector<Point2f> pixPointsCam2;
    for (int k = 0; k < homogPixPointsCam2.size(); k++) {
        Point2f pixPointCam2 = homogToInhomog2D(homogPixPointsCam2[k]);
        pixPointsCam2.push_back(pixPointCam2);
    }
    // 10. draw cylinder points onto image
    cvtColor(cam2Binarized, cam2Binarized, COLOR_GRAY2BGR);
    Mat cam2BinarizedCopy = cam2Binarized.clone();
    for (int l = 0; l < pixPointsCam2.size(); l++) {
        circle(cam2BinarizedCopy, pixPointsCam2[l], 1, Scalar(0, 0, 255), 1, 8);
    }
    // 11. Display image
    imshow("Cam 2 Projection", cam2BinarizedCopy);

    // Finally, projecting into camera 1 frame
    // 12. multiple world cylinder points with cam 1 extrins to get points in cam 1 space [Xc1, Yc1, Zc1]^T
    vector<Mat> cylPointsCam1;
    for (int i = 0; i < homog3DCylPoints.size(); i++) {
        Mat cylPointCam1 = objectPoseCam1 * homog3DCylPoints[i];
        cylPointsCam1.push_back(cylPointCam1);
    }
    // 13. multiply with cam 1 intrins to get homogeneous pixel points [z1x1 z1y1 z1]^T
    vector<Mat> homogPixPointsCam1;
    for (int k = 0; k < cylPointsCam1.size(); k++) {
        Mat homogPixPointCam1 = cam1Intrins * cylPointsCam1[k];
        homogPixPointsCam1.push_back(homogPixPointCam1);
    }
    // 14. convert to inhomogeneous pixel points [x1 y1]^T
    vector<Point2f> pixPointsCam1;
    for (int k = 0; k < homogPixPointsCam1.size(); k++) {
        Point2f pixPointCam1 = homogToInhomog2D(homogPixPointsCam1[k]);
        pixPointsCam1.push_back(pixPointCam1);
    }
    // 15. draw cylinder points onto image
    cvtColor(cam1Binarized, cam1Binarized, COLOR_GRAY2BGR);
    Mat cam1BinarizedCopy = cam1Binarized.clone();
    for (int l = 0; l < pixPointsCam1.size(); l++) {
        circle(cam1BinarizedCopy, pixPointsCam1[l], 1, Scalar(0, 0, 255), 1, 8);
    }
    // 16. Display image
    imshow("Cam 1 Projection", cam1BinarizedCopy);
    
    // Pose estimation optimisation done in Matlab (these are the returned extrinsics)
    Mat matlabRot = (Mat_<float>(3, 3) <<   0.9996,    0.0185, - 0.0218,    
                                            - 0.0213, - 0.0288, - 0.9994,   
                                            - 0.0191,    0.9994, - 0.0284  );
    //matlabRot = matlabRotCorrect * matlabRot;
    Mat matlabTrans = (Mat_<float>(3, 1) << 5.6683, 38.3774, 253.3361);
    Mat matlabRt = Mat(4, 4, CV_32F);
    hconcat(matlabRot, matlabTrans, matlabRt);
    vconcat(matlabRt, bottomRow, matlabRt);
    waitKey();

    return 0;
}
 