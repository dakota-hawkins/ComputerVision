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
    open_full = cv::imread("../BinaryImages/open-bw-partial.png",
                           cv::IMREAD_GRAYSCALE);
    // cv::Mat b_img = binarize_image(open_full); // this is probably losing the info.

    std::vector<std::pair<int, int> > n8_test = clockwise_n8(3, 3, 5, 5);
    print_vector_of_pairs(n8_test);
    cv::Mat b_img;
    cv::threshold(open_full, b_img, 5, 1, cv::THRESH_BINARY_INV);
    cv::Mat eroded;
    cv::Mat mask = cv::Mat::ones(2, 2, CV_8UC1);
    erosion(b_img, eroded, mask);
    cv::Mat labelled = recursive_label(eroded);
    cv::Mat colored_full = color_labels(labelled);
    cv::Mat eroded_color = color_labels(eroded);
    cv::namedWindow("open-bw-full", cv::WINDOW_AUTOSIZE);
    cv::imshow("open-bw-full", open_full);
    cv::namedWindow("colored_full", cv::WINDOW_AUTOSIZE);
    //cv::setMouseCallback("colored_full", mouse_callback, &colored_full);
    cv::imshow("colored_full", colored_full);
    cv::namedWindow("colored-eroded", cv::WINDOW_AUTOSIZE);
    cv::imshow("colored-eroded", eroded_color);
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
                // cout << "pixel @ " << row << ", " << col << endl;
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


void find_boundary(cv::Mat& src, cv::Mat& dst) {
    using ::std::cerr;
    using ::std::endl;
    using ::std::cout;

    if (src.empty()) {
        cerr << "No source image provided." << endl;
    }
    if (dst.empty()) {
        dst = cv::Mat::zeros(src.size(), CV_16UC1);
    }
    if (dst.size() != src.size()) {
        cerr << "Source and destination images are different sizes." << endl;
    }
    std::vector<std::vector<std::pair<int, int> > > boundaries;
    cv::Mat search_img = src.clone();
    for (int row = 0; row < src.rows; row++) {
        for (int col = 0; col < src.cols; col++) {
            int value = search_img.at<uchar>(row, col); // looking for non-zero boundaries. 
            if (value != 0) {
                // location for first boundary pixel
                std::vector<std::pair<int , int> > border; // vector of pixels for border
                std::pair<int, int> c_pxl; // current border pixel
                c_pxl = std::make_pair(row, col);
                border.push_back(c_pxl);


                // Get 4 neighbors to find edge of boundary.
                std::pair<int, int> b_pxl; // background pixel 
                std::vector<std::pair<int, int> > background_neighbors;
                background_neighbors = get_n4(row, col, src.rows, src.cols);

                // find background pixel 
                for (int i = 0; i < background_neighbors.size(); i++) {
                    b_pxl = background_neighbors[i];   
                    if (search_img.at<uchar>(b_pxl.first, b_pxl.second) == 0) {
                        break;
                    }
                }
                
                // start search for boundary at current pixel. 
                int search_row = border[0].first;
                int start_col = border[0].second;
                bool begin_search = true;
                while ((begin_search) || c_pxl != border[0]) {
                    begin_search = false; // set to false once inside while loop.
                    
                    std::vector<std::pair<int, int> > n8; // vector containing neighbor pixels in clockwise order.
                    
                    // get clockwise 8 neighborhood
                    n8 = clockwise_n8(c_pxl.first, c_pxl.second, src.rows, src.cols);
                    
                    // need to start search at background pixel `b_pxl`, find `b_pxl`. 
                    int search_idx = 0; // index of neighbor vector
                    while (n8[search_idx] != b_pxl) {
                        search_idx++;
                    }
                    
                    // search starting at `b_pxl`.
                    bool next_pxl = false; // whether boundary pixel has been found.
                    while (! next_pxl) {
                        int n8_row = n8[search_idx % n8.size()].first;
                        int n8_col = n8[search_idx % n8.size()].second;
                        next_pxl = (search_img.at<uchar>(n8_row, n8_col) == value);
                        if (next_pxl) {
                            // assign current pixel to matched value
                            c_pxl = n8[search_idx % n8.size()];
                            b_pxl = n8[(search_idx -1) % n8.size()];
                            border.push_back(c_pxl);
                        }
                        search_idx++;
                    }
                }
                draw_border(dst, border, value);
                draw_border(search_img, border, 0);
                boundaries.push_back(border);
                // draw onto dst
                // remove from search img.

            }
        }
    }
}

void draw_border(cv::Mat& dst, std::vector<std::pair<int, int> > border, int value) {
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


std::vector<std::pair<int, int> > clockwise_n8(int c_row, int c_col, int n_rows, int n_cols) {
    std::vector<std::pair<int, int> > n8;
    bool left = c_col - 1 >= 0;
    bool up = c_row - 1 >= 0;
    bool right = c_col + 1 < n_cols;
    bool down = c_row + 1 < n_rows;

    if (left) {
        n8.push_back(std::pair<int, int>(c_row, c_col - 1));
    }
    if (left && up) {
        n8.push_back(std::pair<int, int>(c_row - 1, c_col - 1));
    }
    if (up) {
        n8.push_back(std::pair<int, int>(c_row - 1, c_col));
    }
    if (up && right) {
        n8.push_back(std::pair<int, int>(c_row - 1, c_col + 1));
    }
    if (right) {
        n8.push_back(std::pair<int, int>(c_row, c_col + 1));
    }
    if (down && right) {
        n8.push_back(std::pair<int, int>(c_row + 1, c_col + 1));
    }
    if (down) {
        n8.push_back(std::pair<int, int>(c_row + 1, c_col));
    }
    if (down && left) {
        n8.push_back(std::pair<int, int>(c_row + 1, c_col - 1));
    }
    return n8;
}

void print_vector_of_pairs(std::vector<std::pair<int, int> > v_of_p) {
    using ::std::cout;
    using ::std::endl;
    for (int i = 0; i < v_of_p.size(); i++) {
        cout << "(" << v_of_p[i].first << ", " << v_of_p[i].second << ")" << endl;
    }
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
