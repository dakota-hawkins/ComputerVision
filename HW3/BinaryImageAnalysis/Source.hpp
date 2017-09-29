// Standard library imports
#include <map>
#include <set>
#include <string>
#include <vector>
#include <iostream>

// OpenCv library
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

// Function declarations

/*
Label connected regions in a binary image.

Performs the Sequential Labelling Algorithm as described in class and here
(https://en.wikipedia.org/wiki/Connected-component_labeling#Sequential_algorithm)

@param b_img(cv::Mat): binary image to label connected components over.

@return (cv::Mat): labelled image with 
*/
cv::Mat sequential_label(cv::Mat& b_img);


/*
Set equivalency between labels.

Used in `sequential_label()` to group connected objects together. Searches
vector of sets for `label1` or `label2` membership. If either are present, add
missing element to set. Create new vector entry otherwise. 

@param class_vec(std::vector<std::set<int> >): vector containing each
    equivalence class represented as a set. Sets contain integer labels for each
    label belonging to the same equivalence class.
@param label1(int): integer label.
@param label2(int): integer label.
@return Void.
*/
void set_equivalencies(std::vector<std::set<int> > & class_vec, int label1, int label2);


/*
Get class label for a provided label.

Searches a vector of sets to look for `label`'s membership. Upon match, returns
vector index + 1 to represent specific class. 

@param class_vec(std::vector<std::set<int> >): vector containing each
    equivalence class represented as a set. Sets contain integer labels for each
    label belonging to the same equivalence class.
@param label(int): integer label.
@return (int): label indicating which equivalency class `label` belongs to. 
*/
int get_equivalency(std::vector<std::set<int> >& class_vec, int label);

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

