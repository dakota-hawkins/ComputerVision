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
