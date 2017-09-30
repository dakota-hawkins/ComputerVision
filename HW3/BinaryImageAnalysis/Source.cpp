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
    using ::std::cout;
    using ::std::endl;
    // cv2::imread("../BinaryImages/open-bw-full.png", "")
    cv::Mat open_full;
    open_full = cv::imread("../BinaryImages/tumor-fold.png",
                           cv::IMREAD_GRAYSCALE);
    //cv::Mat open_full_binary = binarize_image(open_full);
    cv::Mat b_img = binarize_image(open_full);
    cv::Mat test = recursive_label(b_img);
    cout << "before conv: " << int(test.at<uchar>(16, 399)) << endl;
    cout << "b.value: " << int(b_img.at<uchar>(16, 399)) << endl;
    cv::Mat colored_full = color_labels(test);
    cv::namedWindow("open-bw-full", cv::WINDOW_AUTOSIZE);
    cv::imshow("open-bw-full", open_full);
    cv::namedWindow("colored_full", cv::WINDOW_AUTOSIZE);
    //cv::setMouseCallback("colored_full", mouse_callback, &colored_full);
    cv::imshow("colored_full", colored_full);
    cv::waitKey(0);
    return 0;
}

cv::Mat recursive_label(cv::Mat& b_img) {
    using ::std::cout;
    using ::std::endl;

    cv::Mat neg_b = cv::Mat::zeros(b_img.rows, b_img.cols, CV_16SC1);
    std::stack<std::pair<int, int> > pixel_stack;
    int count = 0;
    // swap 1s to -1s
    for (int b_row = 0; b_row < b_img.rows; b_row++) {
        for (int b_col = 0; b_col < b_img.cols; b_col++) {
            if (b_img.at<uchar>(b_row, b_col) == 1) {
                neg_b.at<schar>(b_row, b_col) = -1;
            }
        }
    }

    // find entry to be labelled.

    for (int row = 0; row < neg_b.rows; row++) {
        for (int col = 0; col < neg_b.cols; col++) {
            // push unlabeled pixel onto stack
            // new object discovered
            if (neg_b.at<schar>(row, col) == -1) {
                cout << "pixel @ " << row << ", " << col << endl;
                count++;
                pixel_stack.push(std::make_pair(row, col));
                while (! pixel_stack.empty()) {
                    // retrieve top entry, remove from stack
                    std::pair<int, int> pixel = pixel_stack.top();
                    // cout << "top: " << pixel.first << ", " << pixel.second << endl;
                    pixel_stack.pop();
                    // label current pixel
                    // cout << "old value: " << int(neg_b.at<schar>(pixel.first, pixel.second)) << endl;
                    neg_b.at<schar>(pixel.first, pixel.second) = count;
                    // cout << "value: " << int(neg_b.at<schar>(pixel.first, pixel.second)) << endl;

                    // get all possible pixels in 8 neighbor.
                    std::vector<std::pair<int, int> > neighbors;
                    neighbors = get_neighbors(neg_b, pixel.first, pixel.second);
                    // if (pixel.first == 16) {
                    //     cout << "original: " << pixel.first << ", " << pixel.second << endl;
                    //     cout << "dims = " << neg_b.rows << ", " << neg_b.cols << endl;
                    //     cout << "neighbors:" << endl;
                    //     for (int k = 0; k < neighbors.size(); k++) {
                    //         cout << '\t' << "y = " << neighbors[k].first << ", x = " << neighbors[k].second << endl;
                    //     }
                    //     cout << endl;
                    // }
                    for (int i = 0; i < neighbors.size(); i++) {
                        std::pair<int, int> curr_neighbor = neighbors[i];
                        // label unlabled neighbors, throw on stack to check
                        // other neighbors.
                        if (neg_b.at<schar>(curr_neighbor.first, curr_neighbor.second) == -1) {
                            neg_b.at<schar>(curr_neighbor.first, curr_neighbor.second) = count;
                            pixel_stack.push(curr_neighbor);
                        }
                    }
                }
                // stack is empty when all touching pixels are labeled, look for next object
            }
        }
    }
    std::cout << "labels: " << count << std::endl;
    cout << "value: " << int(neg_b.at<schar>(16, 399)) << endl;
    return neg_b;
}

std::vector<std::pair<int, int> > get_neighbors(cv::Mat& img, int y, int x) {
    // Repsect image boundaries.
    int row_start = std::max(0, y - 1);
    int row_end = std::min(img.rows, y + 1);
    int col_start = std::max(0, x - 1);
    int col_end = std::min(img.cols, x + 1);
    // int row_start = y - 1;
    // int row_end =  y + 1;
    // int col_start =  x - 1;
    // int col_end = x + 1;
    // Instantiate empty vector
    std::vector<std::pair<int, int> > neighbors;
    for (int row = row_start; row < row_end + 1; row++) {
        for (int col = col_start; col < col_end + 1; col++) {
            if (row != y || col != x) {
                neighbors.push_back(std::make_pair(row, col));
            }
        }
    }
    return neighbors;
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
            if (label_img.at<schar>(row, col) != 0) {
                int color_idx = (label_img.at<schar>(row, col)) % 6;
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

void mouse_callback(int event, int  x, int  y, int  flags, void *param){
    if (event == CV_EVENT_LBUTTONDOWN) {
        cv::Mat &colored_full = *((cv::Mat*)(param));
        cv::Vec3b value = colored_full.at<cv::Vec3b>(y, x);
        std::cout << "y = " << y << ", x = " << x;
        std::cout << ", value = (";
        for (int i = 0; i < 2; i++) {
            std::cout << int(value[i]) << ", ";
        }
        std::cout << int(value[3]) << ")\n";
    }
}
