// standard library
#include<iostream>
#include<vector>
#include<string>
#include<dirent.h>
#include<stdio.h>
#include<map>
#include<math.h>

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
Trace a binary image onto another image.

@param src: source image to trace.
@param dst: destination image to write to.
@param value: pixel value to assign over trace.
*/
void trace_binary(cv::Mat & src, cv::Mat & dst, int value);
void trace_binary(cv::Mat & src, cv::Mat & dst, cv::Vec3b value);