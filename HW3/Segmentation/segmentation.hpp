// Standard Library
#include <iostream>
#include <vector>
#include <math.h>

// OpenCV
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


/*
Perform adaptive thresholding on a given image. 

@param img(cv::Mat): image to threshold.
@param dst(cv::Mat): image to write thesholded image to.
@param mask_size(int): length of square mask to use to calculate mean.
@param C(double): value to subtract from neighborhood mean when comparing
    thresholds.

@return void
*/
void adaptive_threshold(cv::Mat& img, cv::Mat& dst, int mask_size, double C);




/*
Perform double thresholding on a given image.

@param img(cv::Mat): image to threshold.
@param dst(cv::Mat): image to write thesholded image to.
@param thresh_1(double): lower bound threshold.
@param thresh_2(double): upper bound threshold.
@return void
*/
void double_threshold(cv::Mat& img, cv::Mat& dst, double thresh_1, double thresh_2);


/*
Perform absolute thresholding on gray values.

@param img(cv::Mat): image to threshold.
@param dst(cv::Mat): image to write resultant binary image to.
@param thresh(double): threshold value.
@param int(val): value to assign matched pixels. 
*/
void simple_threshold(cv::Mat& img, cv::Mat& dst, double thresh, int val);

/*
Get the euclidean distance between two gray values. 

@param val1(int): first gray value to compare.
@param val2(int): second gray value to compare. 

@return (double): euclidean distance between `val1` and `val2`. 
*/
double distance(int val1, int val2);


/*
Get the euclidean distance between two BGR values.

@param val1 (cv::Vec3b): first BGR value
@param val2 (cv::Vec3b): second BGR value.

@return (double): euclidean distance between `val1` and `val2`. 
*/
double distance(cv::Vec3b val1, cv::Vec3b val2);
double distance(cv::Vec3b val1, cv::Scalar val2);


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
