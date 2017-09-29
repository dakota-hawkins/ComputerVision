/*
Dakota Hawkins
CS 585 Homework 3

This program performs binary image analysis.

explanation: 
*/

// import libraries, function declarations, etc. from header file
// no namespaces for clarity
#include "Source.hpp"

// Main function for binary image analysis.
int main(int argc, char * argv[]) {
    // cv2::imread("../BinaryImages/open-bw-full.png", "")
    cv::Mat open_full;
    open_full = cv::imread("../BinaryImages/open-bw-full.png",
                           cv::IMREAD_GRAYSCALE);
    cv::Mat open_full_binary = binarize_image(open_full);
    cv::Mat labeled_full = sequential_label(open_full_binary);
    cv::Mat colored_full = color_labels(labeled_full);
    cv::namedWindow("open-bw-full", cv::WINDOW_AUTOSIZE);
    cv::imshow("open-bw-full", open_full);
    cv::namedWindow("open-full-labeled", cv::WINDOW_AUTOSIZE);
    cv::imshow("open-full-labeled", colored_full);
    cv::waitKey(0);
    return 0;
}

// Sequential labelling algorithm.
cv::Mat sequential_label(cv::Mat& b_img){
    // instantiate `b_img` sized grayscale image of zeros for component labels
    cv::Mat dst = cv::Mat::zeros(b_img.rows, b_img.cols, CV_8UC1);
    
    // used to keep track of labels
    int count = 0;

    // vector of sets for equivalence classes between labels.
    std::vector<std::set<int> > set_vec;

    // Label connected regions    
    for (int row = 0; row < b_img.rows - 1; row++) {
        for (int col = 0; col < b_img.cols - 1; col++) {
            int pixel = b_img.at<uchar>(row + 1, col + 1);
            if (pixel != 0) {
                int diag = dst.at<uchar>(row, col);
                int left = dst.at<uchar>(row + 1, col);
                int up = dst.at<uchar>(row, col + 1);
                // check if pixel connected to upper diagnol
                if (diag != 0) {
                    dst.at<uchar>(row + 1, col + 1) = diag;
                    // next
                } else {
                    if (left != 0) {
                        // check if up is labelled too
                        if (up != 0) {
                            // assign to left label if same
                            if (up == left) {
                                dst.at<uchar>(row + 1, col + 1) = left;
                            // update labels, map pixel to minimum label
                            // map max label to minimum
                            } else {
                                int small = std::min(up, left);
                                int big = std::max(up, left);
                                dst.at<uchar>(row + 1, col + 1) = small;
                                set_equivalencies(set_vec, small, big);
                            }
                        // connect to left if up in not labelled
                        } else {
                            dst.at<uchar>(row + 1, col + 1) = left;
                        }
                    // check up label
                    } else if (up != 0) {
                        // connect pixel to up
                        dst.at<uchar>(row + 1, col + 1) = up;
                    } else {
                        // No connections, create new label.
                        count ++;
                        dst.at<uchar>(row + 1, col + 1) = count;
                    }
                }
            }
        }
    }

    // Map equivalent labels to each other. 
    // only do this if count > 0
    for (int row = 0; row < dst.rows; row++) {
        for (int col = 0; col < dst.cols; col++) {
            if (dst.at<uchar>(row, col) != 0) {
                dst.at<uchar>(row, col) = get_equivalency(set_vec, dst.at<uchar>(row, col));
            }
        }
    }

    return dst;
}

// Generate equivalence classes.
void set_equivalencies(std::vector<std::set<int> >& class_vec, int label1, int label2) {
   for (int i = 0; i < class_vec.size(); i++) {
       // 
       if (class_vec[i].find(label1) != class_vec[i].end()) {
           class_vec[i].insert(label2);
       } else if (class_vec[i].find(label2) != class_vec[i].end()) {
           class_vec[i].insert(label1);
       } else {
           std::set<int> new_set;
           new_set.insert(label1);
           new_set.insert(label2);
           class_vec.push_back(new_set);
       }

   }
}

int get_equivalency(std::vector<std::set<int> >& class_vec, int label) {
    if (class_vec.size() != 1) {
        for (int i = 0; i < class_vec.size(); i++) {
            if (class_vec[i].find(label) != class_vec[i].end()) {
                return i + 1;
            }
        }
    }
    return 1;
}


// convert integer labels to color values. 
cv::Mat color_labels(cv::Mat& label_img) {
    cv::Mat colored = cv::Mat(label_img.rows, label_img.cols, CV_8UC3,
                              cv::Scalar(255, 255, 255));
    // Create map from integers to BGR color values. 
    std::map<int, cv::Vec3b> color_map;  //
    // Vec3b is B, G, R
    cv::Vec3b RED = cv::Vec3b(66, 66, 244);
    cv::Vec3b BLUE = cv::Vec3b(244, 110, 66);
    cv::Vec3b YELLOW = cv::Vec3b(66, 203, 244);
    cv::Vec3b GREEN = cv::Vec3b(45, 140, 40);
    cv::Vec3b ORANGE = cv::Vec3b(27, 103, 196);
    cv::Vec3b PURPLE = cv::Vec3b(196, 27, 128);
    color_map.insert(std::pair<int, cv::Vec3b>(1, RED));
    color_map.insert(std::pair<int, cv::Vec3b>(2, BLUE));
    color_map.insert(std::pair<int, cv::Vec3b>(3, YELLOW));
    color_map.insert(std::pair<int, cv::Vec3b>(4, GREEN));
    color_map.insert(std::pair<int, cv::Vec3b>(5, ORANGE));
    color_map.insert(std::pair<int, cv::Vec3b>(6, PURPLE));

    // iterate over labelled image, match label to (label mod 6) color.
    for (int row = 0; row < colored.rows; row++) {
        for (int col = 0; col < colored.cols; col++) {
            if (label_img.at<uchar>(row, col) != 0) {
                int color_idx = label_img.at<uchar>(row, col) % 6;
                colored.at<cv::Vec3b>(row, col) = color_map[color_idx]; 
            }
        }
    }
    return colored;
}

// Convert grayscale binary image to 0's and 1's
cv::Mat binarize_image(cv::Mat& gray_img) {
    cv::Mat binary_img = cv::Mat::zeros(gray_img.rows, gray_img.cols, CV_8UC1);
    for (int row = 0; row < gray_img.rows; row++) {
        for (int col = 0; col < gray_img.cols; col++) {
            if (gray_img.at<uchar>(row, col) == 0) {
                binary_img.at<uchar>(row, col) = 1;
            } else {
                binary_img.at<uchar>(row, col) = 0;
            }
        }
    }
    return binary_img;
}

