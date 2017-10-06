// standard library
#include<iostream>
#include<vector>
#include<string>
#include <dirent.h>
#include <stdio.h>

// opencv
#include <opencv2/opencv.hpp>

// my stuff
// #include "../BinaryImageAnalysis/banalysis.hpp"
#include "../Segmentation/segmentation.hpp"

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
