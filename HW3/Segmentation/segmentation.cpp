#include "segmentation.hpp"

// int main() {
//     using ::std::cout;
//     using ::std::endl;

//     cv::Mat bats = cv::imread("../BatImages/Gray/CS585Bats-Gray_frame000000750.ppm", CV_8UC1);
//     cv::Mat color_bats = cv::imread("../BatImages/FalseColor/CS585Bats-FalseColor_frame000000750.ppm", CV_32FC3);
//     cv::Mat dst;

//     // For bats:
//     // 45, 100
//     // 45, 150
//     cout << "gray test: " << distance(0, 10) << endl;
//     cv::Vec3b color_1 = cv::Vec3b(255, 0, 0);
//     cv::Vec3b color_2 = cv::Vec3b(0, 255, 0);
//     cv::Scalar color_3 = cv::Scalar(0, 255, 0);
//     cout << "color test: " << distance(color_1, color_2) << endl;
//     cout << "Scalar test: " << distance(color_1, color_3) << endl;
//     cv::Mat double_thresh;
//     double_threshold(bats, double_thresh, 45, 150);
//     adaptive_threshold(bats, dst, 7, 3);
//     cv::namedWindow("Source", cv::WINDOW_NORMAL);
//     cv::namedWindow("Double Threshold", cv::WINDOW_NORMAL);
//     cv::namedWindow("Smoothed", cv::WINDOW_NORMAL);
//     cv::resizeWindow("Source", 512, 512);
//     cv::resizeWindow("Double Threshold", 512, 512);
//     cv::resizeWindow("Smoothed", 512, 512);
//     cv::imshow("Source", bats);
//     cv::imshow("Double Threshold", dst);
//     cv::imshow("Smoothed", double_thresh);
//     cv::waitKey(0);
//     return 0;
// }

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
            if (n_channels = 1) {
                if (img.at<uchar>(row, col) > mask_mean[0] - C) {
                    dst.at<uchar>(row, col) = 255;
                } else {
                    dst.at<uchar>(row, col) = 0;
                }
            }
            else if (n_channels = 3) {
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