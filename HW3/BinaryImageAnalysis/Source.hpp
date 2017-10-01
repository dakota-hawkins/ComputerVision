// Standard library imports
#include <map>
#include <set>
#include <string>
#include <vector>
#include <iostream>
#include <stack>

// OpenCv library
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

// Function declarations

/*
Label connected regions using the stacked recursive algorithm.

@param b_img(cv::Mat): binary image to label connected components over.
*/
cv::Mat recursive_label(cv::Mat& b_img);


/*
Retrieve the 8-neighborhood of a given pixel.
@param img(cv::Mat): image containing pixel of interest.
@param y(int): y coordinate for pixel of interest.
@param x(int): x coordinate for pixel of interest. 
*/
std::vector<std::pair<int, int> > get_neighbors(cv::Mat& img, int y, int x);


/*
Map labeled image subsections to unique colors for display.

@param label_img(cv::Mat): labelled image. Usually labelled by
    `sequential_label()`

@return (cv::Mat): RGB image with colored labels. 
*/
cv::Mat color_labels(cv::Mat& label_img);


/*
Performs erosion over an image given a particular structuring element.

@param b_img(cb__mat): binary image.
@param dst(cv::Mat): output image to write to. 
@param mask(cv::Mat): binary structuring element
@return void
*/
void erosion(cv::Mat& b_img, cv::Mat& dst, cv::Mat& mask);


/*
Erode a sub image using a given structuring element. 

@param sub_image(cv::Mat): (n x n) binary sub-image.
@param mask(cv::Mat): (n x n) binary structuring element.
@param value(int, optional): value to check for. Default is 1. 

@return (int): `value` if all pixels in sub-image equal `value`, 0 otherwise.
*/
int erode(cv::Mat& sub_image, cv::Mat& mask, int value = 1);


/*
Trace boundary of objects within an image.

@param src(cv::Mat): source image.
@param dst(cv::Mat): destination image.

@return void.
*/
void find_boundary(cv::Mat& src, cv::Mat& dst);


/*
Print vector of integer pairs.
*/
void print_vector_of_pairs(std::vector<std::pair<int, int> > v_of_p);


/*
Get N4 of a pixel in clockwise order starting from the West.

Returns the 4-neighbor for a given pixel. Neighbors outside of image boundaries
will be ignored.

@param c_row(int): current row.
@param c_col(int): current column.
@param n_rows(int): total number of rows in image.
@param n_cols(int): total number of columns in image.
*/
std::vector<std::pair<int, int> > get_n4(int c_row, int c_col, int n_rows, int n_cols);

/*
Binarize image by converting 255's to 0 and 0's to 1's. 

In grayscale, a binary image is full of 0's and 255's. However, algorithmically,
we generally like to work with images where background is as 0's and the
region of interest is represented by 1's.

@param gray_img(cv::Mat): binary image in grayscale.
@return (cv::Mat): image made of 0's and 1's. 
*/
cv::Mat binarize_image(cv::Mat& gray_img);


// GUI functions

// Taken from here:
// https://stackoverflow.com/questions/42005809/can-i-get-the-mouse-position-in-opencv-without-a-mouse-event
void mouse_callback(int  event, int  x, int  y, int  flag, void *param);
