#include "part1.hpp"

int main() {
    // open_partial();
    // open_full();
    // open_fist();
    tumor_fold();
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
    std::vector<cv::Mat> skelaton_vec;
    cv::Mat borders = cv::Mat::zeros(label_vec[0].size(), CV_8UC1);
    cout << "labeled objects: " << n_labels << endl;
    std::ofstream out_file;
    out_file.open("open-bw-partial-stats.csv");
    out_file << "Object,Area,Perimeter,Orientation,Circularity,Compactness,Color" << endl;
    std::string color_string[] = {"red", "blue", "yellow", "green", "orange", "purple"};
    //vector<std::string> color_vec;

    for (int i = 0; i < label_vec.size(); i++) {
        cv::Mat colored_full = color_labels(label_vec[i]);
        std::map<std::string, double> stats = calculate_statistics(label_vec[i]);
        // cout << "Label: " << i << " area =  " << stats["area"] << " | circularity = "<< stats["circ"] << " | theta = "<< stats["theta"] << endl;
        if (stats["area"] > 500) {
            filtered_segments.push_back(label_vec[i]);
            cv::Mat mask = cv::Mat::ones(cv::Size(3,3), CV_8UC1);
            cv::Mat dilated, skelaton;
            dilate(label_vec[i], dilated, mask);
            skelatonize(label_vec[i], skelaton);
            skelaton_vec.push_back(skelaton);
            std::vector<std::pair<int, int> > border = find_boundary(dilated);
            draw_border(borders, border, (filtered_segments.size() % 6));
            out_file << filtered_segments.size() << "," << stats["area"] << ",";
            out_file << border.size() << "," << stats["theta"] << ",";
            out_file << stats["circ"] << "," << double(stats["area"])/double(border.size());
            out_file << "," << color_string[filtered_segments.size() % 6];
            out_file << endl;
        }
    }
    out_file.close();
    cout << "Number of segments kept: " << filtered_segments.size() << endl;
    cv::Mat isolated = cv::Mat::zeros(img.size(), CV_8UC1);
    vector_to_img(isolated, filtered_segments);
    cv::Mat skelatons = cv::Mat::zeros(img.size(), CV_8UC1);
    vector_to_img(skelatons, skelaton_vec);
    cv::Mat color_skelatons = color_labels(skelatons);
    cv::Mat color_segments = color_labels(isolated);
    cv::Mat color_borders = color_labels(borders);
    cv::namedWindow("Open Partial");
    cv::namedWindow("Open Partial - Skelatons");
    cv::namedWindow("Open Partial - Segments");
    cv::namedWindow("Open Partial - Borders");
    cv::imshow("Open Partial", img);
    cv::imwrite("open-partial-skelatons.jpg", color_skelatons);
    cv::imwrite("open-partial-segments.jpg", color_segments);
    cv::imwrite("open-partial-borders.jpg", color_borders);
    cv::imshow("Open Partial - Skelatons", color_skelatons);
    cv::imshow("Open Partial - Segments", color_segments);
    cv::imshow("Open Partial - Borders", color_borders);
    cv::waitKey(0);
}


void open_full() {
    using ::std::cout;
    using ::std::endl;

    // read in image
    cv::Mat img = cv::imread("../BinaryImages/open-bw-full.png", CV_8UC1);
    cv::Mat binary;
    // make binary 
    cv::threshold(img, binary, 5, 1, cv::THRESH_BINARY_INV);
    // label image
    cv::Mat labeled;
    int n_labels = recursive_label(binary, labeled);
    std::vector<cv::Mat> label_vec = label_img_to_vector(labeled, n_labels);

    // Filter segments by size, draw borders for filtered segments.
    std::vector<cv::Mat> filtered_segments;
    std::vector<cv::Mat> skelaton_vec;
    cv::Mat borders = cv::Mat::zeros(label_vec[0].size(), CV_8UC1);
    cout << "labeled objects: " << n_labels << endl;
    std::ofstream out_file;
    out_file.open("open-full-stats.csv");
    out_file << "Object,Area,Perimeter,Orientation,Circularity,Compactness,Color" << endl;
    std::string color_string[] = {"red", "blue", "yellow", "green", "orange", "purple"};
    //vector<std::string> color_vec;
    cv::namedWindow("Current Segment");
    for (int i = 0; i < label_vec.size(); i++) {
        // cv::Mat colored_full = color_labels(label_vec[i]);
        // cv::imshow("Current Segment", colored_full);
        std::map<std::string, double> stats = calculate_statistics(label_vec[i]);
        // cout << "Label: " << i << " area =  " << stats["area"] << " | circularity = "<< stats["circ"] << " | theta = "<< stats["theta"] << endl;
        if (stats["area"] > 500) {
            filtered_segments.push_back(label_vec[i]);
            cv::Mat mask = cv::Mat::ones(cv::Size(3,3), CV_8UC1);
            cv::Mat dilated, skelaton;
            dilate(label_vec[i], dilated, mask);
            skelatonize(dilated, skelaton);
            skelaton_vec.push_back(skelaton);
            std::vector<std::pair<int, int> > border = find_boundary(dilated);
            draw_border(borders, border, (filtered_segments.size() % 6));
            out_file << filtered_segments.size() << "," << stats["area"] << ",";
            out_file << border.size() << "," << stats["theta"] << ",";
            out_file << stats["circ"] << "," << double(stats["area"])/double(border.size());
            out_file << "," << color_string[filtered_segments.size() % 6];
            out_file << endl;
        }
    }
    out_file.close();
    cout << "Number of segments kept: " << filtered_segments.size() << endl;
    cv::Mat isolated = cv::Mat::zeros(img.size(), CV_8UC1);
    vector_to_img(isolated, filtered_segments);
    cv::Mat skelatons = cv::Mat::zeros(img.size(), CV_8UC1);
    vector_to_img(skelatons, skelaton_vec);
    cv::Mat color_skelatons = color_labels(skelatons);
    cv::Mat color_segments = color_labels(isolated);
    cv::Mat color_borders = color_labels(borders);
    cv::namedWindow("Open Full");
    cv::namedWindow("Open Full - Skelatons");
    cv::namedWindow("Open Full - Segments");
    cv::namedWindow("Open Full - Borders");
    cv::imshow("Open Full", img);
    cv::imwrite("open-full-skelatons.jpg", color_skelatons);
    cv::imwrite("open-full-segments.jpg", color_segments);
    cv::imwrite("open-full-borders.jpg", color_borders);
    cv::imshow("Open Full - Skelatons", color_skelatons);
    cv::imshow("Open Full - Segments", color_segments);
    cv::imshow("Open Full - Borders", color_borders);
    cv::waitKey(0);
}


void open_fist() {
    using ::std::cout;
    using ::std::endl;

    // read in image
    cv::Mat img = cv::imread("../BinaryImages/open_fist-bw.png", CV_8UC1);
    cv::Mat binary;
    // make binary 
    cv::threshold(img, binary, 5, 1, cv::THRESH_BINARY_INV);
    // label image
    cv::Mat labeled;
    int n_labels = recursive_label(binary, labeled);
    std::vector<cv::Mat> label_vec = label_img_to_vector(labeled, n_labels);

    // Filter segments by size, draw borders for filtered segments.
    std::vector<cv::Mat> filtered_segments;
    std::vector<cv::Mat> skelaton_vec;
    cv::Mat borders = cv::Mat::zeros(label_vec[0].size(), CV_8UC1);
    cout << "labeled objects: " << n_labels << endl;
    std::ofstream out_file;
    out_file.open("open-fist-stats.csv");
    out_file << "Object,Area,Perimeter,Orientation,Circularity,Compactness,Color" << endl;
    std::string color_string[] = {"red", "blue", "yellow", "green", "orange", "purple"};
    //vector<std::string> color_vec;
    cv::namedWindow("Current Segment");
    for (int i = 0; i < label_vec.size(); i++) {
        // cv::Mat colored_full = color_labels(label_vec[i]);
        // cv::imshow("Current Segment", colored_full);
        // cv::waitKey(0);
        std::map<std::string, double> stats = calculate_statistics(label_vec[i]);
        // cout << "Label: " << i << " area =  " << stats["area"] << " | circularity = "<< stats["circ"] << " | theta = "<< stats["theta"] << endl;
        if (stats["area"] > 600) {
            filtered_segments.push_back(label_vec[i]);
            cv::Mat mask = cv::Mat::ones(cv::Size(3,3), CV_8UC1);
            cv::Mat dilated, skelaton;
            dilate(label_vec[i], dilated, mask);
            skelatonize(dilated, skelaton);
            skelaton_vec.push_back(skelaton);
            std::vector<std::pair<int, int> > border = find_boundary(dilated);
            draw_border(borders, border, (filtered_segments.size() % 6));
            out_file << filtered_segments.size() << "," << stats["area"] << ",";
            out_file << border.size() << "," << stats["theta"] << ",";
            out_file << stats["circ"] << "," << double(stats["area"])/double(border.size());
            out_file << "," << color_string[filtered_segments.size() % 6];
            out_file << endl;
        }
    }
    out_file.close();
    cout << "Number of segments kept: " << filtered_segments.size() << endl;
    cv::Mat isolated = cv::Mat::zeros(img.size(), CV_8UC1);
    vector_to_img(isolated, filtered_segments);
    cv::Mat skelatons = cv::Mat::zeros(img.size(), CV_8UC1);
    vector_to_img(skelatons, skelaton_vec);
    cv::Mat color_skelatons = color_labels(skelatons);
    cv::Mat color_segments = color_labels(isolated);
    cv::Mat color_borders = color_labels(borders);
    cv::namedWindow("Open Full");
    cv::namedWindow("Open Full - Skelatons");
    cv::namedWindow("Open Full - Segments");
    cv::namedWindow("Open Full - Borders");
    cv::imshow("Open Full", img);
    cv::imwrite("open-fist-skelatons.jpg", color_skelatons);
    cv::imwrite("open-fist-segments.jpg", color_segments);
    cv::imwrite("open-fist-borders.jpg", color_borders);
    cv::imshow("Open Full - Skelatons", color_skelatons);
    cv::imshow("Open Full - Segments", color_segments);
    cv::imshow("Open Full - Borders", color_borders);
    cv::waitKey(0);
}

void tumor_fold() {
    using ::std::cout;
    using ::std::endl;

    // read in image
    cv::Mat img = cv::imread("../BinaryImages/tumor-fold.png", CV_8UC1);
    cv::Mat binary;
    // make binary 
    cv::threshold(img, binary, 5, 1, cv::THRESH_BINARY_INV);

    cv::Mat e_mask = cv::Mat::ones(cv::Size(5, 5), CV_8UC1);
    cv::Mat d_mask = cv::Mat::ones(cv::Size(3, 3), CV_8UC1);
    cv::Mat eroded;
    cv::namedWindow("tumor");
    cv::imshow("tumor", img);
    erosion(binary, eroded, e_mask);
    cv::Mat eroded_2;
    erosion(eroded, eroded_2, d_mask);

    cv::Mat dilated;
    cv::dilate(eroded_2, dilated, d_mask);
    cv::Mat colored = color_labels(dilated);
    cv::imwrite("tumor-fold-dilated.jpg", colored);
    cv::namedWindow("dilated");
    cv::imshow("dilated", colored);
    cv::waitKey(0);

    // label image
    cv::Mat labeled;
    cout << "Labeling..." << endl;
    int n_labels = recursive_label(dilated, labeled);
    std::vector<cv::Mat> label_vec = label_img_to_vector(labeled, n_labels);
    for (int i = 0; i < label_vec.size(); i++) {
        cv::Mat new_labels;
        int n = recursive_label(label_vec[i], new_labels);
        if (n > 1) {
            std::vector<cv::Mat> new_vec = label_img_to_vector(new_labels, n);
            label_vec[i] = new_vec[0];
            for (int j = 1; j < new_vec.size(); j++) {
                label_vec.push_back(new_vec[j]);
            }
        }
    }


    // Filter segments by size, draw borders for filtered segments.
    std::vector<cv::Mat> filtered_segments;
    std::vector<cv::Mat> skelaton_vec;

    cout << "labeled objects: " << n_labels << endl;
    std::ofstream out_file;
    out_file.open("tumor-fold-stats.csv");
    out_file << "Object,Area,Perimeter,Orientation,Circularity,Compactness,Color" << endl;
    std::string color_string[] = {"red", "blue", "yellow", "green", "orange", "purple"};


    cv::namedWindow("Current Segment");
    // cv::Mat color_seg = color_labels(labeled);
    // cv::imshow("C")

    std::vector<cv::Mat> border_vec;
    for (int i = 0; i < label_vec.size(); i++) {
        // cv::Mat colored_full = color_labels(label_vec[i]);
        // cv::imshow("Current Segment", colored_full);
        // cv::waitKey(0);
        std::map<std::string, double> stats = calculate_statistics(label_vec[i]);
        // cout << "Label: " << i << " area =  " << stats["area"] << " | circularity = "<< stats["circ"] << " | theta = "<< stats["theta"] << endl;
        if (stats["area"] > 200) {
            // color segments
            // cv::imshow("Current Segment", color_labels(label_vec[i]));
            // cv::waitKey(0);
            filtered_segments.push_back(label_vec[i]);
            // find and draw skelatons
            cv::Mat skelaton;
            skelatonize(label_vec[i], skelaton);
            skelaton_vec.push_back(skelaton);
            // find boundaries
            cv::dilate(label_vec[i], dilated, d_mask);
            std::vector<std::pair<int, int> > border = find_boundary(dilated);
            // find and draw borders
            cv::Mat border_mat = cv::Mat::zeros(label_vec[i].size(), CV_8UC1);
            draw_border(border_mat, border, 1);
            border_vec.push_back(border_mat);
            // write to out file.
            out_file << filtered_segments.size() << "," << stats["area"] << ",";
            out_file << border.size() << "," << stats["theta"] << ",";
            out_file << stats["circ"] << "," << double(stats["area"])/double(border.size());
            out_file << "," << color_string[filtered_segments.size() % 6];
            out_file << endl;
        }
    }
    cout << border_vec.size() << endl;
    cout << filtered_segments.size() << endl;
    out_file.close();
    cout << "Number of segments kept: " << filtered_segments.size() << endl;
    cv::Mat isolated = cv::Mat::zeros(img.size(), CV_8UC1);
    vector_to_img(isolated, filtered_segments);
    cv::Mat skelatons = cv::Mat::zeros(img.size(), CV_8UC1);
    vector_to_img(skelatons, skelaton_vec);
    cv::Mat borders = cv::Mat::zeros(img.size(), CV_8UC1);
    vector_to_img(borders, border_vec);
    cv::Mat color_skelatons = color_labels(skelatons);
    cv::Mat color_segments = color_labels(isolated);
    cv::Mat color_borders = color_labels(borders);
    cv::namedWindow("Open Full");
    cv::namedWindow("Open Full - Skelatons");
    cv::namedWindow("Open Full - Segments");
    cv::namedWindow("Open Full - Borders");
    cv::imshow("Open Full", img);
    cv::imwrite("tumor-fold-skelatons.jpg", color_skelatons);
    cv::imwrite("tumor-fold-segments.jpg", color_segments);
    cv::imwrite("tumor-fold-borders.jpg", color_borders);
    cv::imshow("Open Full - Skelatons", color_skelatons);
    cv::imshow("Open Full - Segments", color_segments);
    cv::imshow("Open Full - Borders", color_borders);
    cv::waitKey(0);
}