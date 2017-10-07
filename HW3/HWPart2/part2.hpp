// standard library
#include<iostream>
#include<fstream>
#include<vector>
#include<string>
#include<dirent.h>
#include<stdio.h>
#include<map>
#include<math.h>
#include<sstream>

// opencv
#include <opencv2/opencv.hpp>

// my stuff
#include "/home/dakota/Documents/School/2017-2018/CS585/HW3/BinaryImageAnalysis/banalysis.hpp"
// #include "../Segmentation/segmentation.hpp"

/*!
List all files in a directory.

Adopted from here: https://stackoverflow.com/questions/23212000/get-the-list-of-files-in-a-directory-with-in-a-directory

@param read_dir (string): directory to read.

@return vector<string>: vector containing all files.
*/
std::vector<std::string> list_files(std::string read_dir);

/*!
Analyze bat data.
*/

void analyze_bats();

/*!
Analyze piano dataset.
*/
void analyze_piano();

/*!
Analyze pedestrian dataset.
*/
void analyze_people();


/*!
Trace a binary image onto another image.

@param src: source image to trace.
@param dst: destination image to write to.
@param value: pixel value to assign over trace.
*/
void trace_binary(cv::Mat & src, cv::Mat & dst, int value);
void trace_binary(cv::Mat & src, cv::Mat & dst, cv::Vec3b value);


/*!
Exctract masked regions from a source region.

@param src: image to extract isolated segments from.
@param mask: binary image denoting segments to extract from source image. 0's
    should represent background while non-zero values represent foreground.

@return: image containing the extracted regions of `src`.
*/
cv::Mat extract_image(cv::Mat & src, cv::Mat & mask);


/*!
Detect skin-pigmented pixels using the Peer metric.

@param src: source image.
@param dst: write image.
*/
void skin_detect(cv::Mat& src, cv::Mat& dst);


/*
Calculate the peer metric for a given bgr pixel.

@param bgr: pixel to analyze.
*/
bool peer_metric(cv::Vec3b& bgr);


// find the max of three integers.
int my_max(int a, int b, int c);


// find the minimum of three integers
int my_min(int a, int b, int c);

/*!
Find the average of two images.
*/
void average_img(cv::Mat & img1, cv::Mat & img2, cv::Mat & dst);