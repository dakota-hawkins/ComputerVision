#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <string>

using namespace cv;
using namespace std;

void tint_image_blue(Mat& image);
Mat convert_to_grayscale(Mat& image);
Mat blur_image(Mat& image);
uchar neighborhood_average(Mat& image, int row, int col);
Mat invert_image(Mat& image);

int main() {
	Mat image;  // Matrix object from openCV
	image = imread("dakota_face.jpg", IMREAD_COLOR); // Read the file
	// image = imread("boston.jpg", IMREAD_GRAYSCALE); // Read file as grayscale
	tint_image_blue(image);
	if (image.empty()) { // Check for invalid input
		cout << "Could not open or find the image" << std::endl;
		return 0;
	}
	Mat gray_image = convert_to_grayscale(image);
	Mat blurred_image = blur_image(gray_image);
	for (int i = 0; i < 10; i++) {
		blurred_image = blur_image(blurred_image);
	}
	Mat inverted_image = invert_image(gray_image);
	// namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	// namedWindow("Original Image", WINDOW_AUTOSIZE);
	// imshow("Original Image", gray_image);
	// imshow("Display window", inverted_image);  // Show our image inside it.

	// waitKey(0); // Wait for a keystroke in the window
	imwrite("gray_image.jpg", gray_image);
	imwrite("blurred_image.jpg", blurred_image);
	imwrite("inverted_image.jpg", inverted_image);
	return 0;
}


void tint_image_blue(Mat& image) {  // Passing pointer
	if (image.empty()) {
		cout << "Error: image is unallocated." << endl;
		return;
	}
	// Color channel is B, G, R
	int channels = image.channels();

	for (int row = 0; row < image.rows; row++) {
		unsigned char* row_ptr = image.ptr<unsigned char>(row);  // pointer that points to specific row
		for (int col = 0; col < image.cols; col++) {
			int index = col*channels + 0; // 0 is the blue channel
			row_ptr[index] = 255;
		}
	}
}

Mat convert_to_grayscale(Mat& image) {
	// Check for empty image
	if (image.empty()) {
		cout << "Error: image is unallocated." << endl;
	}
	// If single-channel image, assume grayscale, return.
	int channels = image.channels();
	if (channels == 1) {
		return image;
	}
	
	// instantiate empty image for grayscale 
	Mat gray_image = Mat::zeros(image.rows, image.cols, CV_8UC1);
	for (int row = 0; row < image.rows; row++) {
		for (int col = 0; col < image.cols; col++) {
			// row gives y index, col gives x index
			Vec3b intensity = image.at<Vec3b>(row, col);
			uchar blue = intensity.val[0];
			uchar green = intensity.val[1];
			uchar red = intensity.val[2];
			// set gray intensity to the average color intensity
			gray_image.at<uchar>(row, col) = (blue + green + red) / 3;
		}
	}
	return gray_image;
}

uchar neighborhood_average(Mat& image, int row, int col) {
	int start_row = max(0, row - 1);
	int end_row = min(image.rows, row + 1);
	int start_col = max(0, col - 1);
	int end_col = min(image.cols, col + 1);
	int sum = 0;
	int count = 0;
	for (int y = start_row; y < end_row; y++) {
		for (int x = start_col; x < end_col; x++) {
			sum += image.at<uchar>(y, x);
			count += 1;
		}
	}
	return sum / count;
}

Mat blur_image(Mat& image) {
	Mat blurred_image = Mat::zeros(image.rows, image.cols, CV_8UC1);
	for (int row = 0; row < image.rows; row++) {
		for (int col = 0; col < image.cols; col++) {
			blurred_image.at<uchar>(row, col) = neighborhood_average(image, row, col);
		}
	}
	return(blurred_image);
}

Mat invert_image(Mat& image) {
	Mat inverted_image = Mat::zeros(image.rows, image.cols, CV_8UC1);
	for (int row = 0; row < image.rows; row++) {
		for (int col = 0; col < image.cols; col++) {
			uchar intensity = image.at<uchar>(row, col);
			inverted_image.at<uchar>(row, col) = abs(255 - intensity);
		}
	}
	return(inverted_image);
}

// void threshold_grayscale(Mat& image, int threshold) {
// 	if (image.empty()) {
// 		cout << "Error: image is unallocated." << endl;
// 		return;
// 	}
// 	int channels = image.channels();
// 	if (channels > 1) {
// 		cout << "Error: image is not grayscale." << endl;
// 		return;
// 	}
// 	for (int row = 0; row < image.rows; row++) {
// 		for (int col = 0; col < image.cols; col++) {
// 			int linear_index = row * image.cols * channels + col * channels + 0;
// 			if (image.data[linear_index] > threshold) {
// 				image.data[linear_index] = 1;
// 			} else {
// 				image.data[linear_index] = 0;
// 			}
// 		}
// 	}
// }



// linearIndex = row * image.cols * channels + col * channels + 0
// image.data[linearIndex] = 255;