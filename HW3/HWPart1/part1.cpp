#include "part1.hpp"

int main() {
    open_partial();
    return 0;
}

void open_partial() {
    using ::std::cout;
    using ::std::endl;

    // read in image
    cv::Mat img = cv::imread("../BinaryImages/open-bw-partial.png", CV_8UC1);
    cv::Mat binary;
    // make binary 
    cv::threshold(img, binary, 5, 1, cv::THRESH_BINARY_INV);
    // label image
    cv::Mat labeled;
    int n_labels = recursive_label(binary, labeled);
    std::vector<cv::Mat> label_vec = label_img_to_vector(labeled, n_labels);

    // Filter segments by size, draw borders for filtered segments.
    std::vector<cv::Mat> filtered_segments;
    cv::Mat borders = cv::Mat::zeros(label_vec[0].size(), CV_8UC1);
    cout << "labeled objects: " << n_labels << endl;
    for (int i = 0; i < label_vec.size(); i++) {
        cv::Mat colored_full = color_labels(label_vec[i]);
        std::map<std::string, double> stats = calculate_statistics(label_vec[i]);
        // cout << "Label: " << i << " area =  " << stats["area"] << " | circularity = "<< stats["circ"] << " | theta = "<< stats["theta"] << endl;
        if (stats["area"] > 500) {
            filtered_segments.push_back(label_vec[i]);
            cv::Mat mask = cv::Mat::ones(cv::Size(3,3), CV_8UC1);
            cv::Mat dilated;
            dilate(label_vec[i], dilated, mask);
            std::vector<std::pair<int, int> > border = find_boundary(dilated);
            draw_border(borders, border, i);
        }
    }
    cout << "Number of segments kept: " << filtered_segments.size() << endl;
    cv::Mat isolated = cv::Mat::zeros(img.size(), CV_8UC1);
    vector_to_img(isolated, filtered_segments);
    cv::Mat color_segments = color_labels(isolated);
    cv::Mat color_borders = color_labels(borders);
    cv::namedWindow("Open Partial");
    cv::namedWindow("Open Partial - Binary");
    cv::namedWindow("Open Partial - Segments");
    cv::namedWindow("Open Partial - Borders");
    cv::imshow("Open Partial", img);
    cv::imshow("Open Partial - Binary", binary);
    cv::imshow("Open Partial - Segments", color_segments);
    cv::imshow("Open Partial - Borders", color_borders);
    cv::waitKey(0);
}