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
vector index + 1 to represent specific class. If element does not belong to a
set, return 0.

@param class_vec(std::vector<std::set<int> >): vector containing each
    equivalence class represented as a set. Sets contain integer labels for each
    label belonging to the same equivalence class.
@param label(int): integer label.
@return (int): label indicating which equivalency class `label` belongs to. 
*/
int get_equivalency(std::vector<std::set<int> >& class_vec, int label);


/*
Merge two sets in a vector of sets together.

@param class_vec(std::vector<std::set<int> >): vector containing each
    equivalence class represented as a set. Sets contain integer labels for each
    label belonging to the same equivalence class.
@param set1_idx(int): vector index for the first set.
@param set2_idx(int): vector index for the second set. 
*/
void merge_classes(std::vector<std::set<int> >& class_vec,
                   int set1_idx, int set2_idx);


/*
print a set of integers.

@param int_set(std::set<int>): set of integers to print.
*/
void print_set(std::set<int> int_set);
void inline print_set(std::set<int> int_set) {
    using ::std::cout;
    std::set<int>::iterator it = int_set.begin();
    cout << "{";
    for (it; it != int_set.end(); it++) {
        cout << *it << ", ";
    }
    cout << "}\n";
}

/*
Add new class to vector of equivlanecy classes.

@param class_vec(std::vector<std::set<int> >): vector containing each
    equivalence class represented as a set. Sets contain integer labels for each
    label belonging to the same equivalence class
@param elements(int array): array of new values to add to set.
@param n(int): number of elements being added. 
*/
void add_class(std::vector<std::set<int> >& class_vec, int elements[], int n);

/*
Print vector of sets to terminal.

@param class_vec(std::vector<std::set<int> >): vector containing each
    equivalence class represented as a set. Sets contain integer labels for each
    label belonging to the same equivalence class.
*/
void print_equivalence_classes(std::vector<std::set<int> >& class_vec);


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
