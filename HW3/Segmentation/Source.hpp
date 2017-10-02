// Standard Library
#include <iostream>

// OpenCV

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


/*
Perform iterative thresholding on a given image.

@param img(cv::Mat): image to threshold.
@param dst(cv::Mat): image to write thesholded image to.

@return void
*/
void iterative_threshold(cv::Mat& img, cv::Mat& dst);


/*
Perform double thresholding on a given image.

@param img(cv::Mat): image to threshold.
@param dst(cv::Mat): image to write thesholded image to.

@return void
*/
void double_threshold(cv::Mat& img, cv::Mat& dst);