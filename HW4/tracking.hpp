// Standard Library Imports
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <math.h>
#include <map>

// OpenCV
#include <opencv2/opencv.hpp>

// BOOST
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string_regex.hpp>
#include <boost/regex.hpp>

// Hungarian
#include "./hungarian-algorithm-cpp-master/Hungarian.h"

//Previous Work
#include "../HW3/BinaryImageAnalysis/banalysis.cpp"

/*!
Convert a .csv file to an image file.

Used to convert bat segmentations into images.

@param csv_file: path to comma separated file representing image segmentations.

@return 8-bit image of mapped csv values. 
*/
cv::Mat csv_to_img(const std::string& csv_file);

/*!
Read a .txt file containing ordered file names.

@param name_file: path to .txt file containing ordered file names. Each line
should represent a unique file. 

@return vector of strings containing file names listed in `name_file`. 
*/
std::vector<std::string> get_file_names(const std::string& name_file);

/*!
Retrieve a vector of images from a vector of file names.

@param file_list: vector of file names.
@param ext: type of file being passed (e.g. ".jpg", ".csv").

@return A vector of images.
*/
std::vector<cv::Mat> file_list_to_data_list(std::vector<std::string> file_list,
                                            const std::string& ext);

/*!
Write vector of images to file.

@param images: vector of images to write.
@param file_names: location to write files to. 

@return void
*/
void img_vec_to_file(std::vector<cv::Mat> images, std::vector<std::string> file_names);


/*!
Create an edmonds adjaceny matrix for bipartite object matching.

@param t0_centers: center predictions from the previous time step.
@param t1_centers: center predictions from current time step.

@return void
*/
cv::Mat edmonds_matrix(std::vector<cv::Point> t0_centers, std::vector<cv::Point> t1_centers);

/*!
Calculate edge length for two given centers.

@param center1: first center.
@param center2: second center.

@return distance between the two centers. 
*/
double edge_length(cv::Point center1, cv::Point center2);

/*!
Create a cost matrix between assignments.
*/
std::vector<std::vector<double> > cost_matrix(std::vector<cv::Point> t0_centers, std::vector<cv::Point> t1_centers);
/*!
Perform the hungarian algorithm over an edmonds matrix.

@param edmonds: edmonds adjancency matrix where each edge is represents the
    distance between centers. Created from edmonds_matrix().

@return map linking t0 centers to t1 centers.
*/
std::map<int, int> hungarian(cv::Mat& edmonds);

/*
Initialize cost vector u during hungarian.
*/
std::vector<double> initialize_s_to_t(cv::Mat& edmonds);

/*
Initialize cost vector v during hungarian.
*/
std::vector<double> initialize_t_to_s(cv::Mat& edmonds, std::vector<double> s_to_t);

/*
Initialize mates during hungarian.
*/
std::map<std::pair<int, int>, std::pair<int, int> > initialize_mates(cv::Mat& residual);