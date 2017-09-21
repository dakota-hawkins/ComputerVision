/*	Source.cpp
*	CS585 Image and Video Computing Fall 2017
*	Homework 2
*	--------------
*	This program detects the following hand gestures from a video
* feed:
*   1: Open hand.
*   2: Fist
*   3: Thumbs up.
*   4: Waving
*	--------------
*/

//opencv libraries
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
//C++ standard libraries
#include <iostream>
#include <vector>
#include <math.h>
#include <utility>
#include <fstream>

using namespace cv;
using namespace std;

// Load in reference images
// Templates for a fist
Mat FIST_TEMPLATE_1 = imread("../Templates/FistTemplate1Smaller.jpg", IMREAD_GRAYSCALE);
Mat FIST_TEMPLATE_2 = imread("../Templates/FistTemplate2Smaller.jpg", IMREAD_GRAYSCALE);
Mat FIST_TEMPLATE_3 = imread("../Templates/FistTemplate3Smaller.jpg", IMREAD_GRAYSCALE);
Mat FIST_TEMPLATE_4 = imread("../Templates/FistTemplate4Smaller.jpg", IMREAD_GRAYSCALE);

// Templates for a hand
Mat HAND_TEMPLATE_1 = imread("../Templates/HandTemplate1Smaller.jpg", IMREAD_GRAYSCALE);
Mat HAND_TEMPLATE_2 = imread("../Templates/HandTemplate2Smaller.jpg", IMREAD_GRAYSCALE);
Mat HAND_TEMPLATE_3 = imread("../Templates/HandTemplate3Smaller.jpg", IMREAD_GRAYSCALE);

// Templates for thumbs up
Mat THUMBS_TEMPLATE_1 = imread("../Templates/ThumbsUpTemplate1Smaller.jpg", IMREAD_GRAYSCALE);
Mat THUMBS_TEMPLATE_2 = imread("../Templates/ThumbsUpTemplate2Smaller.jpg", IMREAD_GRAYSCALE);
Mat THUMBS_TEMPLATE_3 = imread("../Templates/ThumbsUpTemplate3Smaller.jpg", IMREAD_GRAYSCALE);

// State Displays
Mat HAND_DISPLAY = imread("../Templates/HandDisplay.jpg", IMREAD_GRAYSCALE);
Mat FIST_DISPLAY = imread("../Templates/FistDisplay.jpg", IMREAD_GRAYSCALE);
Mat THUMBS_DISPLAY = imread("../Templates/ThumbsUpDisplay.jpg", IMREAD_GRAYSCALE);
Mat WAVE_DISPLAY = imread("../Templates/WaveDisplay.jpg", IMREAD_GRAYSCALE);

//function declarations

/**
Function that returns the maximum of 3 integers
@param a first integer
@param b second integer
@param c third integer
*/
int myMax(int a, int b, int c);

/**
Function that returns the minimum of 3 integers
@param a first integer
@param b second integer
@param c third integer
*/
int myMin(int a, int b, int c);

/**
Function that detects whether a pixel belongs to the skin based on RGB values
@param src The source color image
@param dst The destination grayscale image where skin pixels are colored white and the rest are colored black
*/
void mySkinDetect(Mat& src, Mat& dst);

/**
Function that does frame differencing between the current frame and the previous frame
@param src The current color image
@param prev The previous color image
@param dst The destination grayscale image where pixels are colored white if the corresponding pixel intensities in the current
and previous image are not the same
*/
void myFrameDifferencing(Mat& prev, Mat& curr, Mat& dst);

/**
Function that accumulates the frame differences for a certain number of pairs of frames
@param mh Vector of frame difference images
@param dst The destination grayscale image to store the accumulation of the frame difference images
*/
void myMotionEnergy(vector<Mat> mh, Mat& dst);

// Dakota Functions

// Dakota Functions

/**
Convert three-channeled images to grayscale using the average intensity.
*/
Mat convert_to_grayscale(Mat& image);

/**
Function to determine whether an BGR color vector matches the Peer metric for
skin color.

@param bgr: vector denoting Blue, Green, Red color pixel.
*/
bool peer_metric(Vec3b& bgr);

/**
Function to estimate the euclidean distance between two RGB color pixels.
*/
int euclidean_color_difference(Vec3b& cur_color, Vec3b& past_color);


/**
Function to create a vector denoting an image pyramid ordered largest to
smallest.
@param img (cv2::Mat) source image used to create image pyramid.
@param n_layers (int) number of layers in the pyramid to create.
@return (vector<Mat>) vector containing downsized images.
*/
vector<Mat> create_image_pyramids(Mat& img, int n_layers=2);

/**
Find a rough boundary box for a binary image using x and y projections.
@param binary image(cv2::Mat): binary image to search.
@param dst(cv2::Mat): matrix to hold boundary box data.
*/
vector<int> boundary_box(Mat& binary_image);

/**
Find zeros in a matrix.
@param src(cv2::Mat): matrix to find zeros in.
@return vector<tuple>: vector of tupels containg row, col coordinates for zeros.
*/
vector<pair<int, int> > find_zeros(Mat& src);

/**
Find boundaries from a projection.
@param projection(cv2::Mat): axis projection of image.
*/
vector<pair<int, int> > find_boundaries(Mat& projection);

/**
Search over image for template matching using NCC


@param src (cv2::Mat): matrix image to compare to template.
@param template_img (cv2::Mat): matrix image of template.
@param ncc_cutoff(double, optional): threshold value to determine whether the
    template exists within the image.

@ return (bool): whether the template image exists within the parent image.
*/
bool match_template(Mat& src, Mat& template_img, double ncc_cutoff = 0.7);


/**
Calculate normalized correlations coefficient (NCC) between two matrices of the
same shape.


@param mat1 (cv2::Mat): first (m x n) matrix.
@param mat2 (cv2::Mat): second (m x n) matrix.

@return (double): calculated NCC.
*/
double calculate_ncc(Mat& mat1, Mat& mat2);

/**
Find gesture in frame.
*/
string find_gesture(Mat& src, pair<double, double>& prev_c_mass);

/**
Display text images of gesture matches.
*/
void display_match(string match);

/**
Calculate center of mass for binary images.
@return pair<double, double> x_bar, y_bar.
*/
pair<double, double> center_of_mass(Mat& binary_image);

/**
Calculate velocity from to center of mass measurements.
*/
pair<double, double> velocity(pair<double, double> com_t0,
                              pair<double, double> com_t1);

int main() {
  VideoCapture cap(0);
  // if not successful, exit program
  if (!cap.isOpened()) {
    cout << "Cannot open the video cam" << endl;
    return -1;
  }

  //create a window called "MyVideoFrame0"
  namedWindow("MyVideo0", WINDOW_AUTOSIZE);
  Mat frame0;

  // read a new frame from video
  bool bSuccess0 = cap.read(frame0);

  //if not successful, break loop
  if (!bSuccess0)
  {
    cout << "Cannot read a frame from video stream" << endl;
  }

  //show the frame in "MyVideo" window
  imshow("MyVideo0", frame0);
  Mat binary_t0;
  mySkinDetect(frame0, binary_t0);
  pair <double, double> curr_c_mass = center_of_mass(binary_t0);
  while (1) {
    // read a new frame from video
    Mat frame;
    bool bSuccess = cap.read(frame);
    imshow("MyVideo0", frame);
    //if not successful, break loop
    if (!bSuccess)
    {
      cout << "Cannot read a frame from video stream" << endl;
      break;
    }

    // Put analysis here

    string gesture = "none";
    display_match(gesture);
    // downsample image to create image pyramid.
    // Image pyramid is ordered largest to smallest.
    const int N_LAYERS = 4;
    vector<Mat> image_pyramid = create_image_pyramids(frame, N_LAYERS);
    int count = N_LAYERS - 1;
    // Update center of mass vector
    while (count > -1 && gesture == "none") {
        // cout << "count: " << count << endl;
        gesture = find_gesture(image_pyramid[count], curr_c_mass);
        display_match(gesture);
        count--;
      }
    // End of analysis

    frame0 = frame;
    //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
    if (waitKey(30) == 27)
    {
      cout << "esc key is pressed by user" << endl;
      break;
    }

  }
  cap.release();
  return 0;
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

vector<Mat> create_image_pyramids(Mat& img, int n_layers) {
    vector<Mat> image_pyramid;
    image_pyramid.push_back(img);
    for (int i = 0; i < n_layers - 1; i++) {
        Mat tmp;
        pyrDown(image_pyramid[i], tmp, Size(image_pyramid[i].cols/2,
                                            image_pyramid[i].rows/2));
        image_pyramid.push_back(tmp);
    }
    return image_pyramid;
}


//Function that returns the maximum of 3 integers
int myMax(int a, int b, int c) {
	int m = a;
	(void)((m < b) && (m = b));
	(void)((m < c) && (m = c));
	return m;
}

//Function that returns the minimum of 3 integers
int myMin(int a, int b, int c) {
	int m = a;
	(void)((m > b) && (m = b));
	(void)((m > c) && (m = c));
	return m;
}

// Function that detects whether a pixel belongs to skin based on RGB values.
bool peer_metric(Vec3b& bgr) {
    int blue, green, red;
    blue = bgr[0];
    green = bgr[1];
    red = bgr[2];
	if (red > 95 && green > 40 && blue > 20) {
		if (myMax(bgr[0], bgr[1], bgr[2]) -  myMin(bgr[0], bgr[1], bgr[2]) > 15) {
			if (abs(bgr[2] - bgr[1]) > 15 && bgr[2] > bgr[1] && bgr[2] > bgr[0]) {
				return true;
			}
        }
    }
	return false; Mat FIST_TEMPLATE_1 = imread("../Templates/FistTemplate1Smaller.jpg", IMREAD_GRAYSCALE);
}


// Function that creates a binary image based on skin detection via RBG values.
void mySkinDetect(Mat& src, Mat& dst) {
	//Surveys of skin color modeling and detection techniques:
	//Vezhnevets, Vladimir, Vassili Sazonov, and Alla Andreeva. "A survey on pixel-based skin color detection techniques." Proc. Graphicon. Vol. 3. 2003.
	//Kakumanu, Praveen, Sokratis Makrogiannis, and Nikolaos Bourbakis. "A survey of skin-color modeling and detection methods." Pattern recognition 40.3 (2007): 1106-1122.

    // Implement Peer method
    if (dst.empty()) {
        dst = Mat::zeros(src.rows, src.cols, CV_8UC1);
    }
	for (int row = 0; row < src.rows; row++) {
		for (int col = 0; col < src.cols; col++) {
            bool skin_color = peer_metric(src.at<Vec3b>(row, col));
            if (skin_color) {
                dst.at<uchar>(row, col) = 255;
            } else {
                dst.at<uchar>(row, col) = 0;
            }
		}
    }
}

// Function to estimate the euclidean distance between two color pixels
int euclidean_color_difference(Vec3b& curr_pixel, Vec3b& past_pixel) {
    int sum_of_square_dif = 0;
    for (int i = 0; i < 3; i ++) {
        sum_of_square_dif += pow((curr_pixel[0] - past_pixel[0]), 2);
    }
    return round(sqrt(sum_of_square_dif));
}

//Function that does frame differencing between the current frame and the previous frame
void myFrameDifferencing(Mat& prev, Mat& curr, Mat& dst) {
	for (int row = 0; row < curr.rows; row++) {
        for (int col = 0; col < curr.cols; col++) {
            Vec3b previous_pixel = prev.at<Vec3b>(row, col);
            Vec3b curr_pixel = curr.at<Vec3b>(row, col);
            dst.at<uchar>(row, col) = euclidean_color_difference(curr_pixel,
                                                                 previous_pixel);
        }
    }
}

//Function that accumulates the frame differences for a certain number of pairs of frames
void myMotionEnergy(vector<Mat> mh, Mat& dst) {
	// Mat first_image = mh[0];
	Mat dif_mat = Mat::zeros(mh[0].rows, mh[0].cols, CV_8UC1);
	for (int t = 1; t < mh.size(); t++) {
		// Find difference
		myFrameDifferencing(mh[t - 1], mh[t], dif_mat);
		// Threshold difference
		for (int row = 0; row < dst.rows; row++) {
			for (int col = 0; col < dst.cols; col++) {
				if (dif_mat.at<uchar>(row, col) > 10) {
					dst.at<uchar>(row, col) = 255;
				}
			}
		}
	}
}


vector<pair<int, int> > find_zeros(Mat& src) {
    vector<pair<int, int> > zero_idx;
    for (int row = 0; row < src.rows; row++) {
        for (int col = 0; col < src.cols; col++) {
            if (src.at<uchar>(row, col) == 0) {
                zero_idx.push_back(make_pair(row, col));
            }
        }
    }
    return zero_idx;
}


vector<pair<int, int> > find_boundaries(Mat& projection) {
    vector<pair<int, int> > range_vec;
    bool in_segment = false;
    pair<int, int> range_pair;
    for (int i = 0; i < projection.rows; i++) {
        if (projection.at<uchar>(i, 0) != 0 && ! in_segment) {
            range_pair.first = i;
            in_segment = true;
        } else if (projection.at<uchar>(i, 0) == 0 && in_segment) {
            range_pair.second = i - 1;
            in_segment = false;
            range_vec.push_back(range_pair);
        } else if (i == projection.rows - 1 && in_segment) {
            range_pair.second = i;
            in_segment = false;
            range_vec.push_back(range_pair);
        }
    }
    return range_vec;
}

vector<int> boundary_box(Mat& binary_image) {
    // in binary image, 255 = 1, 0 = 0
    // calculate projections using vector multiplication to sum
    Mat float_img = binary_image.clone();  // convert to float for matrix mult
    float_img.convertTo(float_img, CV_32FC1);
    Mat t_binary = Mat::zeros(binary_image.cols, binary_image.rows, CV_32FC1);
    transpose(float_img, t_binary);
    Mat y_projection = float_img * Mat::ones(binary_image.cols, 1, CV_32FC1);
    Mat x_projection = t_binary * Mat::ones(t_binary.cols, 1, CV_32FC1);

    // convert back to uchar
    y_projection.convertTo(y_projection, CV_8UC1);
    x_projection.convertTo(x_projection, CV_8UC1);

    // Find boundaries based on zeros in projections.
    vector<pair<int, int> > y_boundaries = find_boundaries(y_projection);
    vector<pair<int, int> > x_boundaries = find_boundaries(x_projection);

    // find the largest region by area covered
    int area = 0;
    int box_y1, box_y2, box_x1, box_x2;
    for (int i = 0; i < y_boundaries.size(); i++) {
        int y1 = y_boundaries[i].second;
        int y0 = y_boundaries[i].first;
        for (int j = 0; j < x_boundaries.size(); j++) {
            int x1 = x_boundaries[j].second;
            int x0 = x_boundaries[j].first;
            int new_area = (y1 - y0) * (x1 - x0);
            if (new_area > area) {
                box_y1 = y0;
                box_y2 = y1;
                box_x1 = x0;
                box_x2 = x1;
                area = new_area;
            }
        }
    }
    // cout << y_projection << endl;
    // cout << "Y boundaries: " << endl;
    // cout << "cols: " <<  binary_image.cols << ", rows: " << binary_image.rows << endl;
    for (int k = 0; k < y_boundaries.size(); k++) {
        pair<int, int> each = y_boundaries[k];
        // cout << '(' << each.first << ", " << each.second << ')' << endl;
    }
    // cout << "X boundaries: " << endl;
    for (int k = 0; k < x_boundaries.size(); k++) {
        pair<int, int> each = x_boundaries[k];
        // cout << '(' << each.first << ", " << each.second << ')' << endl;
    }
    // some form of overflow?
    int arr[] = {box_y1, box_y2, box_x1, box_x2};
    // for (int i = 0; i < 4; i++) {
    //     cout << arr[i] << ", ";
    // }
    // cout << endl;
    vector<int> vec (arr, arr+sizeof(arr) / sizeof(arr[0]));
    return vec;
}

double calculate_ncc(Mat& mat1, Mat& mat2) {
    if (mat1.rows != mat2.rows || mat1.cols != mat2.cols) {
        cout << "Matrices must be the same size.";
        return -5;
    }
    // Mat m1_flat = mat1.reshape(1, 1);
    // Mat m2_flat = mat2.reshape(1, 1);
    Scalar m1_bar, m2_bar, m1_dev, m2_dev;
    meanStdDev(mat1, m1_bar, m1_dev);
    meanStdDev(mat2, m2_bar, m2_dev);
    double ncc;
    // Size length = m1_flat.size();
    for (int row = 0; row < mat1.rows; row++) {
        for (int col = 0; col < mat1.cols; col++) {
            double x = mat1.at<uchar>(row, col);
            double y = mat2.at<uchar>(row, col);
            ncc += (x - m1_bar[0])*(y - m2_bar[0])/(m1_dev[0]*m2_dev[0]);
        }
    }
    ncc = abs(ncc / (mat1.rows*mat1.cols));
    // cout << "Calculated ncc: " << ncc << endl;
    return ncc;
}


bool match_template(Mat& src, Mat& template_img, double ncc_cutoff) {
    // assumption that template image is smaller than src image may be off.
    int window_length = template_img.rows;
    int window_width = template_img.cols;
    if (window_length > src.rows) {
        window_length = src.rows;
    }
    if (window_width > src.cols) {
        window_width = src.cols;
    }
    Mat search_image;
    for (int row = 0; row + window_length < src.rows; row++) {
        for (int col = 0; col + window_width < src.cols; col++) {
            search_image = src(Range(row, row + window_length),
                               Range(col, col + window_width));
            double ncc = calculate_ncc(search_image, template_img);
            // cout << "Returned ncc: " << ncc << endl;
            if (ncc >= ncc_cutoff) {
                return true;
            }
        }
    }
    return false;
}

string find_gesture(Mat& src, pair<double, double>& prev_c_mass) {
    Mat gray_src;
    if (src.channels() > 1) {
        cvtColor(src, gray_src, CV_BGR2GRAY);
    } else {
        gray_src = src.clone();
    }
    Mat binary_src = Mat::zeros(gray_src.rows, gray_src.cols, CV_8UC1);
    mySkinDetect(src, binary_src);
    pair <double, double> curr_c_mass = center_of_mass(binary_src);
    // cout << "Binary Image made." << endl;
    vector<int> limits = boundary_box(binary_src);
    // cout << "Limits calculated:" << endl;
    // for (int i = 0; i < 4; i++) {
    //     cout << limits[i] << ", ";
    // }
    // cout << endl;
    Mat gray_src_box = gray_src(Range(limits[0], limits[1]),
                                Range(limits[2], limits[3]));
    // cout << "Image sub-selected." << endl;
    double length = (double)limits[1] - (double)limits[0];
    double width = (double)limits[3] - (double)limits[2];
    double l_w_ratio = length/width;
    pair <double, double> change_in_dir;
    change_in_dir = velocity(prev_c_mass, curr_c_mass);
    if (abs(change_in_dir.first) > 2) {
      prev_c_mass = curr_c_mass;
      return "wave";
    } else if (l_w_ratio < 0.95) {
        return "fist";
    }
    else if (l_w_ratio > 1.4) {
        return "hand";
    }
    else if (match_template(gray_src_box, THUMBS_TEMPLATE_1), 0.7) {
        return "thumbs";
    }
    prev_c_mass = curr_c_mass;
    return "none";
}

void display_match(string match) {
    namedWindow("Gesture Label", WINDOW_AUTOSIZE);
    Mat gesture_label;
    if (match == "fist") {
      gesture_label = imread("../Templates/FistDisplay.jpg");
    } else if (match == "hand") {
      gesture_label = imread("../Templates/HandDisplay.jpg");
    } else if (match == "thumbs") {
      gesture_label = imread("../Templates/ThumbsUpDisplay.jpg");
    } else if (match == "wave") {
      gesture_label = imread("../Templates/WaveDisplay.jpg");
    } else {
      gesture_label = imread("../Templates/NoDisplay.jpg");
    }
    imshow("Gesture Label", gesture_label);
}

pair<double, double> center_of_mass(Mat& binary_image) {
    pair<double, double> com;
    Moments m = moments(binary_image, true);
    com.first = m.m10/m.m00;
    com.second = m.m01/m.m00;
    return com;
}

pair<double, double> velocity(pair<double, double> com_t0, pair<double, double> com_t1) {
    pair<double, double> delta;
    delta.first = com_t1.first - com_t0.first; // ∆x
    delta.second = com_t1.second - com_t0.second; // ∆y
    return delta;
}
