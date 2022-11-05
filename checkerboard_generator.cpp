#include <opencv2/opencv.hpp>
#include <cmath>
#include <windows.h>

using namespace cv;
using namespace std;

int main() {
	// --------------- Generating a checkerboard --------------- //
	// Device specs: 
	// - iPad: Pixels 2360x1640, ppi 264
	// - Galaxy S10 3040x1440, ppi 551

	int pixel_width = 1800;
	int pixel_height = 1200;

	// Number of squares (With white border surrounding)
	int block_length = 200;	// Pixels per length
	int rows = floor(pixel_height / block_length);	// Round down
	int cols = floor(pixel_width / block_length);		// Round down

	// Boundary cases
	int bound_pixels_horiz = (pixel_width - cols * block_length) / 2;
	int bound_pixels_vert = (pixel_height - rows * block_length) / 2;

	// Checkerboard matrix (Board is white, insert black blocks)
	Mat board = Mat::ones(pixel_height, pixel_width, CV_8UC1) * 255;

	// Inserting black blocks onto white board
	for (int i = bound_pixels_vert; i < (pixel_height - bound_pixels_vert); i += block_length) {
		if (((i - bound_pixels_vert) / block_length) % 2 == 0) {
			for (int j = bound_pixels_horiz; j < (pixel_width - bound_pixels_horiz); j += 2 * block_length) {
				for (int k = 0; k < block_length; k++) {
					for (int l = 0; l < block_length; l++) {
						board.at<uchar>(i + k, j + l) = 0;
					}
				}
			}
		}
		else {
			for (int j = bound_pixels_horiz + block_length; j < (pixel_width - bound_pixels_horiz); j += 2 * block_length) {
				for (int k = 0; k < block_length; k++) {
					for (int l = 0; l < block_length; l++) {
						board.at<uchar>(i + k, j + l) = 0;
					}
				}
			}
		}
	}

	// Displaying checkerboard
	string checker_window_name = "Checkerboard Window";
	cv::namedWindow(checker_window_name, WINDOW_KEEPRATIO);
	cv::imshow(checker_window_name, board);
	cv::waitKey();

	// Saving the image
	string img_name = "Checkerboards/checkerboard_" + to_string(pixel_width) + "x" + to_string(pixel_height) + "_" + to_string(cols) + "x" + to_string(rows) + "_blocks_" + to_string(block_length) + "_pixels.jpg";
	imwrite(img_name, board);

	return 0;
}