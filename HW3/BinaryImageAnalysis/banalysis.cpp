/*
Dakota Hawkins
CS 585 Homework 3

This program performs binary image analysis and segmentation.

explanation:
*/

// import libraries, function declarations, etc. from header file
// no namespaces for clarity
#include "banalysis.hpp"

// // Main function for binary image analysis.
// int main(int argc, char * argv[]) {
//     using ::std::cout;
//     using ::std::endl;
//     // cv2::imread("../BinaryImages/open-bw-full.png", "")
//     cv::Mat open_full;
//     open_full = cv::imread("../BinaryImages/open-bw-partial.png",
//                            cv::IMREAD_GRAYSCALE);
//     cv::Mat b_img;
//     cv::Mat skel_test;
//     cv::threshold(open_full, b_img, 5, 1, cv::THRESH_BINARY_INV);
//     skelatonize(b_img, skel_test);
//     cv::Mat color_skel = color_labels(skel_test);
//     std::map<std::string, double> stats = calculate_statistics(b_img);
//     cout << "Area: " << stats["area"] << endl;
//     cout << "Theta: " << stats["theta"] << endl;
//     cout << "Circularity: " << stats["circ"] << endl;
//     cv::Mat eroded;
//     cv::Mat mask = cv::Mat::ones(2, 2, CV_8UC1);
//     erosion(b_img, eroded, mask);
//     cv::Mat labelled; 
//     int n_labels = recursive_label(eroded, labelled);
//     std::vector<cv::Mat> label_vec = label_img_to_vector(labelled, n_labels);
//     cv::Mat all_boarders = cv::Mat::zeros(label_vec[0].size(), CV_8UC1);
//     cv::Mat oi;
//     vector_to_img(oi, label_vec);
//     cout << "labeled objects: " << n_labels << endl;
//     for (int i = 0; i < label_vec.size(); i++) {
//         // std::vector<std::pair<int, int> > border = find_boundary(label_vec[i]);
//         // draw_border(all_boarders, border, 1);
//         cv::Mat colored_full = color_labels(label_vec[i]);
//         cout << i << endl;
//         cv::namedWindow("colored_full", cv::WINDOW_AUTOSIZE);
//         cv::imshow("colored_full", colored_full);
//         cv::waitKey(0);
//     }
//     cv::Mat colored_full = color_labels(label_vec[4]);
//     cv::Mat eroded_color = color_labels(eroded);
//     cv::Mat borders = cv::Mat::zeros(labelled.size(), CV_8UC1);
//     std::vector<std::pair<int, int> > border = find_boundary(label_vec[1]);
//     draw_border(borders, border, 1);
//     cv::Mat colored_borders = color_labels(all_boarders);
//     cv::namedWindow("open-bw-full", cv::WINDOW_AUTOSIZE);
//     cv::imshow("open-bw-full", open_full);
//     cv::namedWindow("colored_full", cv::WINDOW_AUTOSIZE);
//     //cv::setMouseCallback("colored_full", mouse_callback, &colored_full);
//     cv::imshow("colored_full", colored_full);
//     cv::namedWindow("colored-eroded", cv::WINDOW_AUTOSIZE);
//     cv::imshow("colored-eroded", eroded_color);
//     cv::namedWindow("colored-borders", cv::WINDOW_AUTOSIZE);
//     cv::imshow("colored-borders", colored_borders);
//     cv::namedWindow("skelaton", cv::WINDOW_AUTOSIZE);
//     cv::imshow("skelaton", color_skel);
//     cv::waitKey(0);
//     return 0;
// }


int recursive_label(cv::Mat& b_img, cv::Mat& dst) {
    using ::std::cout;
    using ::std::endl;

    if (b_img.empty()) {
        std::cerr << "Binary image is empty.";
    }

    if (dst.empty()) {
        dst = cv::Mat::zeros(b_img.rows, b_img.cols, CV_16UC1);

    }

    if (dst.size() != b_img.size()) {
        std::cerr << "Binary image and labelled image must have equal dimensions.";
    }

    cv::Mat neg_b = cv::Mat::zeros(b_img.rows, b_img.cols, CV_32SC1);
    int count = 0;
    // swap 1s to -1s
    for (int b_row = 0; b_row < b_img.rows; b_row++) {
        for (int b_col = 0; b_col < b_img.cols; b_col++) {
            if (b_img.at<uchar>(b_row, b_col) == 1) {
                neg_b.at<schar>(b_row, b_col) = -1;
            }
        }
    }

    // find entries to be labelled.
    for (int row = 0; row < neg_b.rows; row++) {
        for (int col = 0; col < neg_b.cols; col++) {
            // push unlabeled pixel onto stack
            // new object discovered
            
            if (neg_b.at<schar>(row, col) == -1) {
                count++;
                std::stack<std::pair<int, int> > pixel_stack;
                pixel_stack.push(std::make_pair(row, col));
                while (! pixel_stack.empty()) {
                    // retrieve top entry, remove from stack
                    std::pair<int, int> pixel = pixel_stack.top();
                    pixel_stack.pop();
                    // label current pixel
                    neg_b.at<schar>(pixel.first, pixel.second) = 1;

                    // get all possible pixels in 8 neighbor.
                    std::vector<std::pair<int, int> > neighbors;
                    neighbors = get_neighbors(neg_b, pixel.first, pixel.second);
                    // std::set<std::pair<int, int> > visited;
                    for (int i = 0; i < neighbors.size(); i++) {
                        std::pair<int, int> curr_neighbor = neighbors[i];
                        // label unlabled neighbors, throw on stack to check
                        // other neighbors.
                        if (neg_b.at<schar>(curr_neighbor.first, curr_neighbor.second) == -1) {
                            neg_b.at<schar>(curr_neighbor.first, curr_neighbor.second) = 1;
                            dst.at<uchar>(curr_neighbor.first, curr_neighbor.second) = count;
                            pixel_stack.push(curr_neighbor);
                        }
                    }
                }
                // stack is empty when all touching pixels are labeled, look for next object
            }
        }
    }

    return count;
}


std::vector<std::pair<int, int> > get_neighbors(cv::Mat& img, int y, int x) {
    // Repsect image boundaries.
    int row_start = std::max(0, y - 1);
    int row_end = std::min(img.rows, y + 1);
    int col_start = std::max(0, x - 1);
    int col_end = std::min(img.cols, x + 1);
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


void erosion(cv::Mat& b_img, cv::Mat& dst, cv::Mat& mask) {
    using ::std::cerr;
    using ::std::cout;
    using ::std::endl;

    if (b_img.empty()) {
        cerr << "No image provided." << endl;
    }

    if (dst.empty()) {
        // check this
        dst = b_img.clone();
    }

    if (b_img.size() != dst.size()) {
        cerr << "Source image and destination image must be the same dimensions.";
    }

    if (mask.empty()) {
        cerr << "No structuring element provided." << endl;
    }

    if (mask.rows > b_img.rows || mask.cols > b_img.cols) {
        cerr << "Structuring element too large.";
    }

    // ignoring boundaries on bottom
    for (int row = 0; row < b_img.rows - mask.rows + 1; row++) {
        // ignoring boundaries on the right
        for (int col = 0; col < b_img.cols - mask.cols + 1; col++) {
            if (b_img.at<schar>(row, col) != 0) {
                cv::Mat selection = b_img(cv::Range(row, row + mask.rows),
                                          cv::Range(col, col + mask.cols));
                dst.at<schar>(row, col) = erode(selection, mask, b_img.at<schar>(row, col));
            } else {
                dst.at<schar>(row, col) = 0;
            }


        }

    }
}


// Performs erosions over a single sub-image. Assumes schar.
int erode(cv::Mat& sub_image, cv::Mat& mask, int value) {
    if (sub_image.size() != mask.size()) {
        std::cerr << "Sub-image and mask must be the same size.\n";
    }
    for (int row = 0; row < sub_image.rows; row++) {
        for (int col = 0; col < sub_image.cols; col++) {
            if (sub_image.at<schar>(row, col) == 0 &&  mask.at<uchar>(row, col) != 0) {
                return 0;
            }
            if (sub_image.at<schar>(row, col) !=0 && mask.at<uchar>(row, col) == 0) {
                return 0;
            }
            // if (sub_image.at<schar>(row, col) != mask.at<schar>(row, col)) {
            //     // std::cout << "pixel @ " <<  row  << ", " << col << std::endl;
            //     // std::cout << "sub-image val: " << int(sub_image.at<schar>(row, col)) << std::endl;
            //     // std::cout << "mask val: " << int(mask.at<schar>(row, col)) << std::endl;
            //     return 0;
            // }
        }
    }
    return sub_image.at<uchar>(0, 0);

}


std::vector<std::pair<int , int> > find_boundary(cv::Mat& src) {
    using ::std::cerr;
    using ::std::endl;
    using ::std::cout;

    if (src.empty()) {
        cerr << "No source image provided." << endl;
    }

    cv::Mat img = cv::Mat::zeros(src.size(), CV_8UC1);
    std::vector<std::pair<int , int> > border; // vector of pixels for border
    cv::Mat search_img = src.clone();
    bool one_boundary = false;
    for (int row = 0; row < src.rows; row++) {
        for (int col = 0; col < src.cols; col++) {
            int value = search_img.at<uchar>(row, col); // looking for non-zero boundaries.
            if (value != 0 && ! one_boundary) {
                cout << "starting boundary search." << endl;
                // location for first boundary pixel
                std::pair<int, int> c_pxl; // current border pixel
                c_pxl = std::make_pair(row, col);
                border.push_back(c_pxl);


                // instantiate vector of pixels and pixel values for neighbors
                std::vector<std::pair<std::pair<int, int>, int> > n8;
                
                // Get neighbors to find edge of boundary.
                n8 = clockwise_n8(search_img, c_pxl.first, c_pxl.second);
                // background_neighbors = get_n4(row, col, src.rows, src.cols);

                // find background pixel, only look at n4.
                std::pair<int, int> b_pxl;
                for (int i = 0; i < n8.size(); i += 2) {
                    if (n8[i].second == 0) {
                        b_pxl = n8[i].first;
                        break;
                    }
                }

                // start search for boundary at current pixel.
                int search_row = border[0].first;
                int start_col = border[0].second;
                bool begin_search = true;
                while ((begin_search) || c_pxl != border[0]) {
                    begin_search = false; // set to false once inside while loop.

                    // vector containing neighbor pixels in clockwise order.
                    std::vector<std::pair<std::pair<int, int>, int> > n8; 

                    // get clockwise 8 neighborhood, vector of pairs with pixel
                    // locations and values. 
                    n8 = clockwise_n8(search_img, c_pxl.first, c_pxl.second);

                    // need to start search at background pixel `b_pxl`, find `b_pxl`.
                    int search_idx = 0; // index of neighbor vector
                    while (n8[search_idx].first != b_pxl) {
                        search_idx++;
                    }

                    // search starting at `b_pxl`.
                    bool next_pxl = false; // whether boundary pixel has been found.
                    while (! next_pxl) {
                        std::pair<int, int> s_pxl = n8[search_idx % n8.size()].first;
                        int s_value =  n8[search_idx % n8.size()].second;
                        next_pxl = (s_value == value);
                        if (next_pxl) {
                            // assign current pixel to matched value
                            c_pxl = s_pxl;
                            b_pxl = n8[(search_idx -1) % n8.size()].first;
                            border.push_back(c_pxl);
                        }
                        search_idx++;
                    }
                }
                one_boundary = true;
            }
            
        }
    }
    return border;
}

void draw_border(cv::Mat& dst, std::vector<std::pair<int, int> > border, int value) {
    if (value == 0) {
        value = 1; 
    }
    for (int i = 0; i < border.size(); i++) {
        dst.at<uchar>(border[i].first, border[i].second) = value;
    }
}

// Return clockwise n4 of a pixel starting from the west.
std::vector<std::pair<int, int> > get_n4(int c_row, int c_col, int n_rows, int n_cols) {
    std::vector<std::pair<int, int> > n4;
    if (c_col - 1 >= 0) {
        n4.push_back(std::pair<int, int>(c_row, c_col - 1));
    }
    if (c_row - 1 >= 0) {
        n4.push_back(std::pair<int, int>(c_row - 1, c_col));
    }
    if (c_col + 1 < n_cols) {
        n4.push_back(std::pair<int, int>(c_row, c_col + 1));
    }
    if (c_row + 1 < n_rows) {
        n4.push_back(std::pair<int, int>(c_row + 1, c_col));
    }
    return n4;
}


std::vector<std::pair<std::pair<int, int>, int> > clockwise_n8(cv::Mat& img, int c_row, int c_col) {
    std::vector<std::pair<std::pair<int, int>, int> > n8;
    bool left = c_col - 1 >= 0;
    bool up = c_row - 1 >= 0;
    bool right = c_col + 1 < img.cols;
    bool down = c_row + 1 < img.rows;
    std::pair<int, int> coords;
    std::pair<std::pair<int, int>, int> pixel;

    // West neighbor
    pixel.first = std::make_pair(c_row, c_col - 1);
    pixel.second = 0;
    if (left) {
        pixel.second = img.at<uchar>(c_row, c_col - 1);
    }
    n8.push_back(pixel); 

    // Northwest neighbor
    pixel.first = std::make_pair(c_row -1, c_col - 1);
    pixel.second = 0;
    if (left && up) {
        pixel.second = img.at<uchar>(c_row - 1, c_col - 1);
    }
    n8.push_back(pixel);
    
    // North neighbor
    pixel.first = std::make_pair(c_row -1 , c_col);
    pixel.second = 0;
    if (up) {
        pixel.second = img.at<uchar>(c_row - 1, c_col);
    }
    n8.push_back(pixel);

    // Northeast neighbor
    pixel.first = std::make_pair(c_row - 1, c_col + 1);
    pixel.second = 0;
    if (up && right) {
        pixel.second = img.at<uchar>(c_row - 1, c_col + 1);
    }
    n8.push_back(pixel);

    // East neighbor
    pixel.first = std::make_pair(c_row, c_col + 1);
    pixel.second = 0;
    if (right) {
        pixel.second = img.at<uchar>(c_row, c_col + 1);
    }
    n8.push_back(pixel);

    // Southeast neighbor
    pixel.first = std::make_pair(c_row + 1, c_col + 1);
    pixel.second = 0;
    if (down && right) {
        pixel.second = img.at<uchar>(c_row + 1, c_col + 1);
    }
    n8.push_back(pixel);

    // South neighbor
    pixel.first = std::make_pair(c_row + 1, c_col);
    pixel.second = 0;
    if (down) {
        pixel.second = img.at<uchar>(c_row + 1, c_col);
    }
    n8.push_back(pixel);

    // Southeast neighbor
    pixel.first = std::make_pair(c_row + 1, c_col -1);
    pixel.second = 0;
    if (down && left) {
        pixel.second = img.at<uchar>(c_row + 1, c_col - 1);
    }
    return n8;
}

void print_vector_of_pairs(std::vector<std::pair<std::pair<int, int>, int> > v_of_p) {
    using ::std::cout;
    using ::std::endl;
    
    for (int i = 0; i < v_of_p.size(); i++) {
        std::pair<int, int> pixel = v_of_p[i].first;
        int value = v_of_p[i].second;
        cout << "(" << pixel.first << ", " << pixel.second << ") = " << value << endl;
    }
}

std::map<std::string, double> calculate_statistics(cv::Mat& img) {
    std::map<std::string, double> stats;
    cv::Moments moments = cv::moments(img, true);
    // Calculate area
    stats["area"] = moments.m00;

    // Calculate orientation
    double x_bar = moments.m10/moments.m00;
    double y_bar = moments.m01/moments.m00;
    stats["x_bar"] = x_bar;
    stats["y_bar"] = y_bar;
    stats["theta"] = 0.5 * std::atan((2.0 * moments.mu11) / (moments.mu20 - moments.mu02));

    // Calculate e_min and e_max => circularity
    double first_term = (moments.m20 + moments.m02) / (moments.m11);
    double second_term = (moments.m20 - moments.m02) / (moments.m11);
    double denom = std::sqrt(std::pow(moments.m20 - moments.m02, 2) +
                             std::pow(moments.m11, 2));
    double third_term = (moments.m20 - moments.m02) / denom;
    double fourth_term = moments.m11/2.0 * (moments.m11/denom);
    double e_min = first_term - second_term * third_term - fourth_term;
    double e_max = first_term + second_term * third_term + fourth_term;
    stats["circ"] = e_min/e_max;

    return stats;

}

void skelatonize(cv::Mat& src, cv::Mat& skelaton) {
    using ::std::cerr;
    using ::std::cout;
    using ::std::endl;

    if (skelaton.empty()) {
        skelaton = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);
    }
    cv::Mat dist_mat;
    cv::distanceTransform(src, dist_mat, CV_DIST_L1, 3);
    dist_mat.convertTo(dist_mat, CV_8UC1); // weird
    
    for (int row = 0; row < src.rows; row++) {
        for (int col = 0; col < src.cols; col++) {
            if (src.at<uchar>(row, col) != 0) {
                std::vector<std::pair<int, int> > c_n4 = get_n4(row, col, src.rows, src.cols);
                double max_distance = 0;
                std::pair<int, int> c_pxl;
                for (int i = 0 ; i < c_n4.size(); i++) {
                    c_pxl = c_n4[i];
                    double dist = dist_mat.at<uchar>(c_pxl.first, c_pxl.second);
                    if (dist > max_distance) {
                        max_distance = dist_mat.at<uchar>(c_pxl.first, c_pxl.second);
                    }
                }
                if (dist_mat.at<uchar>(row, col) >= max_distance) {
                    skelaton.at<uchar>(row, col) = src.at<uchar>(row, col);
                } else {
                    skelaton.at<uchar>(row, col) = 0;
                }
            }
        }
    }
    
}

std::vector<cv::Mat> label_img_to_vector(cv::Mat& label_img, int n_labels) {
    std::vector<cv::Mat> mat_vec;
    for (int label = 1; label < n_labels + 1; label++) {
        cv::Mat c_img = cv::Mat::zeros(label_img.size(), CV_8UC1);
        for (int row = 0; row < label_img.rows; row++) {
            for (int col = 0; col < label_img.cols; col++) {
                if (label_img.at<uchar>(row, col) == label) {
                    c_img.at<uchar>(row, col) = 1;
                }
            }
        }
        mat_vec.push_back(c_img);
    }
    return mat_vec;
}

void vector_to_img(cv::Mat dst, std::vector<cv::Mat> img_vec) {
    if (dst.empty()) {
        dst = cv::Mat::zeros(img_vec[0].size(), CV_8UC1);
    }
    if (dst.size() != img_vec[0].size()) {
        std::cerr << "Error: dimensions do not match.\n";
    }
    for (int i = 0; i < img_vec.size(); i++) {
        for (int row = 0; row < img_vec[0].rows; row++) {
            for (int col = 0; col < img_vec[0].cols; col++){
                if (img_vec[i].at<uchar>(row, col) != 0) {
                    dst.at<uchar>(row, col) = i + 1;
                }
            }
        }
    }

}

// convert integer labels to color values.
cv::Mat color_labels(cv::Mat& label_img) {
    cv::Mat colored = cv::Mat(label_img.rows, label_img.cols, CV_8UC3,
                              cv::Scalar(255, 255, 255));
    // Create map from integers to BGR color values.
    std::vector<cv::Vec3b> color_vec;  //
    // Vec3b is B, G, R
    color_vec.push_back(cv::Vec3b(66, 66, 244)); // red
    color_vec.push_back(cv::Vec3b(244, 110, 66));  // blue
    color_vec.push_back(cv::Vec3b(66, 203, 244)); // yellow
    color_vec.push_back(cv::Vec3b(45, 140, 40));  // green
    color_vec.push_back(cv::Vec3b(27, 103, 196));  // orange
    color_vec.push_back(cv::Vec3b(196, 27, 128));  // purple

    // iterate over labelled image, match label to (label mod 6) color.
    for (int row = 0; row < colored.rows; row++) {
        for (int col = 0; col < colored.cols; col++) {
            if (label_img.at<uchar>(row, col) != 0) {
                int color_idx = (label_img.at<uchar>(row, col)) % color_vec.size();
                // std::cout << "r = " << row << " c = " << col << colored.size() << std::endl;
                if (color_idx < color_vec.size()) {
                    colored.at<cv::Vec3b>(row, col) = color_vec[color_idx];
                } else {
                    colored.at<cv::Vec3b>(row, col) = cv::Vec3b(0, 0, 0);
                }
                
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


// --------------------------Segmentation Functions-----------------------------


void adaptive_threshold(cv::Mat& img, cv::Mat& dst, int mask_size, double C) {
    int n_channels = img.channels();
    using ::std::cerr;
    using ::std::cout;
    using ::std::endl;

    if (img.empty()) {
        cerr << "Error: cannot access image data" << endl;
        return; 
    }

    if (dst.empty()) {
        dst = cv::Mat(img.size(), CV_8UC1);
    }

    if (dst.channels() > 1) {
        cerr << "Error: Expected single channel image for `dst`." << endl;
    }

    for (int row = 0; row < img.rows ; row++) {
        int row_start = std::max(0, row - mask_size/2);
        int row_end = std::min(img.rows, row + mask_size/2);

        for (int col = 0; col < img.cols; col++) {
            int col_start = std::max(0, col - mask_size/2);
            int col_end = std::min(img.cols, col + mask_size/2);

            cv::Mat mask = img(cv::Range(row_start, row_end),
                               cv::Range(col_start, col_end));
            cv::Scalar mask_mean = cv::mean(mask);
            // cout <<"mean: " << mask_mean << endl;
            if (n_channels == 1) {
                if (img.at<uchar>(row, col) > mask_mean[0] - C) {
                    dst.at<uchar>(row, col) = 255;
                } else {
                    dst.at<uchar>(row, col) = 0;
                }
            }
            else if (n_channels == 3) {
                double dif_mag = distance(dst.at<cv::Vec3b>(row, col), mask_mean);
                double avg_diff = 0;
                for (int y = 0; y < mask.rows; y++) {
                    for (int x = 0; x < mask.cols; x ++) {
                        avg_diff += distance(dst.at<cv::Vec3b>(x, y), mask_mean);
                    }
                }
                if (dif_mag > avg_diff/(mask.rows * mask.cols) - C) {
                    dst.at<uchar>(row, col) = 255;
                } else {
                    dst.at<uchar>(row, col) = 0;
                }
                
            }
        }
    }
}

void double_threshold(cv::Mat& img, cv::Mat& dst, double thresh_1, double thresh_2) {
    using ::std::cerr;
    using ::std::cout;
    using ::std::endl;

    if (img.empty()) {
        cerr << "Error: cannot access image data." << endl;
        return;
    }
    if (img.channels() > 1) {
        cerr << "Error: double thresholding only on grayscale images." << endl;
    }
    if (dst.empty()) {
        dst = cv::Mat::zeros(img.size(), CV_8UC1);
    }
    if (dst.size() != img.size()) {
        cerr << "Error: source and destination images must have the same dimensions." << endl;
        return;
    }

    // Threshold image into three regions.
    // R1 = p(x, y) s.t. p(x, y) < thresh_1
    // R2 = p(x, y) s.t. thresh_1 <= p(x, y) < thresh_2
    // R3 = p(x, y) s.t. p(x, y) >= thresh_2 
    for (int row = 0; row < img.rows; row++) {
        for (int col = 0; col < img.cols; col++) {
            int value = img.at<uchar>(row, col);
            if (value < thresh_1) {
                dst.at<uchar>(row, col) = 0;
            } else if (value >= thresh_1 && value < thresh_2) {
                dst.at<uchar>(row, col) = 2;
            } else {
                dst.at<uchar>(row, col) = 255;
            }
        }
    }

    // Merge pixels from R2 into R1 if the pixel has neighbors in R1. 
    // Repeat until no pixels are re-assigned. 
    bool movement = true;
    while (movement) {
        movement = false;
        for (int row = 0; row < dst.rows; row++) {
            for (int col = 0; col < dst.cols; col++) {
                if (dst.at<uchar>(row, col) == 2) {
                    std::vector<std::pair<int, int> > n4;
                    n4 = get_n4(row, col, dst.rows, dst.cols);
                    for (int i = 0; i < n4.size(); i++) {
                        if (dst.at<uchar>(n4[i].first, n4[i].second) == 0) {
                            dst.at<uchar>(row, col) = 0;
                            movement = true;
                            break;
                        }
                    }
                                    
                }
            }
        }
    }

    // Re-assign remaining pixels in R2 to R3. 
    for (int row = 0; row < dst.rows; row++) {
        for (int col = 0; col < dst.cols; col++) {
            if (dst.at<uchar>(row, col) == 2) {
                dst.at<uchar>(row, col) = 255;
            }
        }
    }

    
}

void simple_threshold(cv::Mat& img, cv::Mat& dst, double thresh, int val) {
    using ::std::cerr;
    using ::std::cout;
    using ::std::endl;
    if (img.empty()) {
        cerr << "Error: cannot access image data." << endl;
        return;
    }
    if (dst.empty()) {
        dst = cv::Mat::zeros(img.size(), CV_8UC1);
    }
    if (dst.size() != img.size()) {
        cerr << "Error: source and destination images must have the same dimensions." << endl;
        return;
    }

    for (int row = 0; row < img.rows; row++) {
        for (int col = 0; col < img.cols; col++) {
            if (img.at<uchar>(row, col) > thresh) {
                img.at<uchar>(row, col) = 1;
            } else {
                img.at<uchar>(row, col) = 0;
            }
        }
    }
}

double distance(int val1, int val2) {

    return abs(val1 - val2);
}

double distance(cv::Vec3b val1, cv::Vec3b val2){
    double distance = 0;
    for (int i = 0; i < 3; i++) {
        distance += std::pow((val1[i] - val2[i]), 2);
    }
    return std::sqrt(distance);
}

double distance(cv::Vec3b val1, cv::Scalar val2) {
    double distance = 0;
    for (int i = 0; i < 3; i++) {
        distance += std::pow((val1[i] - val2[i]), 2);
    }
    return std::sqrt(distance);
}