#include "tracking.hpp"

using ::std::cout;
using ::std::cerr;
using ::std::endl;


int main() {

    // std::vector<std::string> files = get_file_names("./BatSegmentation/file_list.txt");
    // std::vector<std::string> sub_files(&files[0], &files[20]);
    // std::vector<cv::Mat> bats = file_list_to_data_list(sub_files, ".csv");
    
    // img_vec_to_file(bats, new_names);
    // cv::Mat data = bats[0];
    // cv::Mat colored_segments = color_labels(data);
    // cv::namedWindow("test", cv::WINDOW_NORMAL);
    // cv::resizeWindow("test", 512, 512);
    // cv::imshow("test", colored_segments);
    // cv::waitKey(0);
    std::vector<cv::Point> p0, p1;
    p0.push_back(cv::Point2f(0, 1));
    p0.push_back(cv::Point2f(2, 4));
    p0.push_back(cv::Point2f(4, 5));
    p1.push_back(cv::Point2f(1, 0));
    p1.push_back(cv::Point2f(3, 3));
    p1.push_back(cv::Point2f(3, 6));
    p1.push_back(cv::Point2f(7, 7));
    cv::Mat edmonds = edmonds_matrix(p0, p1);
    cout << edmonds << endl;
    std::vector<std::vector<double> > cost_m ;//= cost_matrix(p0, p1);
    double row1[4] = {82, 83, 69, 92};
    double row2[4] = {77, 37, 49, 92};
    double row3[4] = {11, 69, 5, 86};
    double row4[4] = {8, 9, 98, 23};
    vector<double> v1(row1, row1 + 4);
    vector<double> v2(row2, row2 + 4);
    vector<double> v3(row3, row3 + 4);
    vector<double> v4(row4, row4 + 4);
    cost_m.push_back(v1);
    cost_m.push_back(v2);
    cost_m.push_back(v3);
    cost_m.push_back(v4);


    for (int i = 0; i < cost_m.size(); i ++) {
        for (int j = 0; j < cost_m[i].size(); j++) {
            cout << cost_m[i][j] << ", ";
        }
        cout << endl;
    }

    HungarianAlgorithm center_assignment;
    std::vector<int> assignments;
    center_assignment.Solve(cost_m, assignments);
    cout << "Assignments: " << endl;
    for (int i = 0; i < assignments.size(); i++) {
        cout << assignments[i] << endl;
    }
    cv::KalmanFilter position_model;

    return 0;
}

cv::Mat csv_to_img(const std::string& csv_file) {
    cv::Mat data = cv::Mat::zeros(cv::Size(1024, 1024), CV_8UC1);
    std::ifstream csv;
    csv.open(csv_file.c_str());
    std::string line, s_value;
    int i_value;
    std::vector<std::string> split_vec;
    int current_row = 0;
    if (csv.is_open()) {
        while (getline(csv, line)) {
            boost::trim(line);
            boost::split(split_vec, line, boost::is_any_of(","));
            for (int i = 0; i < split_vec.size(); i++) {
                s_value = split_vec[i];
                std::stringstream str_to_int(s_value);
                str_to_int >> i_value;
                data.at<uchar>(current_row, i) = i_value;
            }
            current_row++;
        }
    }
    csv.close();
    return data;
}

std::vector<std::string> get_file_names(const std::string& name_file) {
    std::ifstream read_file(name_file.c_str());
    std::vector<std::string> file_vec;
    std::string line;
    while (std::getline(read_file, line)) {
        file_vec.push_back(line);
    }
    return file_vec;
}

std::vector<cv::Mat> file_list_to_data_list(std::vector<std::string> file_list,
    const std::string& ext) {
    std::vector<cv::Mat> img_vec;
    cv::Mat temp;
    if (ext == "csv" || ext == ".csv") {
        for (int i = 0; i < file_list.size(); i++) {
            temp = csv_to_img(file_list[i]);
            img_vec.push_back(temp);
        }
    } else if (ext == "jpg" || ext == ".jpg") {
        for (int i = 0; i < file_list.size(); i++) {
            temp = cv::imread(file_list[i], CV_8UC1);
            img_vec.push_back(temp);
        }
    } else {
        cerr << "Error: unexpected extension." << endl;
    }
    temp.release();
    return img_vec;
}

void img_vec_to_file(std::vector<cv::Mat> images, std::vector<std::string> file_names) {
    for (int i = 0; i < images.size(); i++) {
        cv::imwrite(file_names[i], images[i]);
    }
}

std::vector<std::vector<double> > cost_matrix(std::vector<cv::Point> t0_centers, std::vector<cv::Point> t1_centers) {
    std::vector<std::vector<double> > cost;
    for (int i = 0; i < t0_centers.size(); i++) {
        std::vector<double> new_row;
        for (int j = 0; j < t1_centers.size(); j++) {
            double edge = 1 + edge_length(t0_centers[i], t1_centers[j]);
            new_row.push_back(edge);
        }
        cost.push_back(new_row);
    }
    return cost;
}

cv::Mat edmonds_matrix(std::vector<cv::Point> t0_centers, std::vector<cv::Point> t1_centers) {
    
    // force square adjacency matrix
    int n_rows = t0_centers.size();
    int n_cols = t1_centers.size();
    if (n_rows > n_cols) {
        n_cols += (t0_centers.size() - t1_centers.size());
    } else if (n_cols > n_rows) {
        n_rows += (t1_centers.size() - t0_centers.size());
    }
    cv::Mat edmonds = cv::Mat::zeros(n_rows, n_cols, CV_64FC1);

    // Maybe limit number of edges somehow
    for (int row = 0; row < t0_centers.size(); row++) {
        for (int col = 0; col < t1_centers.size(); col++) {
            // set edge length from c_row to c_col -- force an edge to exist
            // so +1 to all weights.
            edmonds.at<double>(row, col) = 1 + edge_length(t0_centers[row],
                                                            t1_centers[col]);
        }
    }
    return edmonds;
}


double edge_length(cv::Point center1, cv::Point center2) {
    return cv::norm(center1 - center2);
}

std::map<int, int> hungarian(cv::Mat& edmonds) {
    std::map<int, int> mapping; // t0 center to t1 center map
    cv::Mat pairing = cv::Mat::zeros(edmonds.size(), CV_8UC1);
    cv::Mat residual = edmonds.clone();  // clone matrix for residuals

    // Dual initialization
    std::vector<double> s_to_t = initialize_s_to_t(edmonds);
    std::vector<double> t_to_s = initialize_t_to_s(residual, s_to_t);
    

    // remove t_to_s weight from residuals
    for (int i = 0; i < residual.rows; i++) {
        for (int j = 0; j < residual.cols; j++) {
            double diff = residual.at<double>(i, j) - t_to_s[j];
            if (diff > -1) {
                residual.at<double>(i, j) = diff;
            }
        }
    }
    cout << residual << endl;

    // Primal initialization
    std::vector<int> t0_mates, t1_mates, t0_labels, t1_labels, p_vec, pi_vec;
    int inf = std::numeric_limits<int>::infinity();
    for (int i = 0; i < edmonds.rows; i++) {
        // negative values represent nulls
        t0_mates.push_back(-1);
        t1_mates.push_back(-1);
        t0_labels.push_back(-1);
        t1_labels.push_back(-1);
        pi_vec.push_back(-1);
        p_vec.push_back(inf);
    }
    int cardinality = 0;

    for (int i = 0; i < residual.rows; i++) {
        for (int j = 0; j < residual.cols; j++) {
            if (t0_mates[i] < 0 && t1_mates[j] < 0 && residual.at<double>(i, j) == 0) {
                pairing.at<uchar>(i, j) = 1;
                cardinality++;
                t0_mates[i] = j;
                t1_mates[j] = i;
            }
        }
    }
    cout << cardinality;
    // Path Initialization
    std::set<int> t0_label_sets, t1_label_sets;
    for (int i = 0; i < residual.rows; i++) {
        if (t0_mates[i] < 0) {
            t0_labels[i] = i;
            t0_label_sets.insert(i);
        }
        for (int j = 0; j < residual.cols; j++) {
            if (residual.at<double>(i, j) < p_vec[j]) {
                p_vec[j] = residual.at<double>(i, j);
                pi_vec[j] =i ;
            }
        }
    }
    // label propagation
    // 3.2.A
    for (std::set<int>::iterator it = t0_label_sets.begin(); it != t0_label_sets.end(); it++) {
        int k = *it;
        for (int j = 0; j < residual.cols; j++) {
            if (t1_labels[j] < 0 && residual.at<double>(k, j) == 0) {
                t1_labels[j] = k;
                t1_label_sets.insert(k);
            }
        }
    }
    // 3.2.B
    int path;
    for (std::set<int>::iterator it = t1_label_sets.begin(); it != t1_label_sets.end(); it++) {
        int k = *it;
        int mate = t1_mates[k];
        if (mate > -1) {
            if (t1_labels[mate] < 0) {
                t1_labels[mate] = k;
                t1_label_sets.insert(k);
                for (int j = 0; j < residual.cols; j++) {
                    if (residual.at<double>(mate, j) < p_vec[j]) {
                        p_vec[j] = residual.at<double>(mate, j);
                        pi_vec[j] = mate;
                    }
                }
            }
        } else {
            path = k;
        }
    }

    return mapping;
}


std::vector<double> initialize_s_to_t(cv::Mat& edmonds) {
    std::vector<double> s_to_t;
    double inf = std::numeric_limits<double>::infinity();
    for (int i = 0; i < edmonds.rows; i++) {
        double min_c = inf; 
        for (int j = 0; j < edmonds.cols; j++) {
            if (edmonds.at<double>(i, j) < min_c) {
                min_c = edmonds.at<double>(i, j);
            }
        }
        s_to_t.push_back(min_c);
    }
    return s_to_t;
}

std::vector<double> initialize_t_to_s(cv::Mat& edmonds, std::vector<double> s_to_t) {
    std::vector<double> t_to_s;
    double inf = std::numeric_limits<double>::infinity();
    for (int j = 0; j < edmonds.cols; j++) {
        double min_c = inf;
        for (int i = 0; i < edmonds.cols; i++) {
            edmonds.at<double>(i, j) -= s_to_t[i];
            if (edmonds.at<double>(i, j) < min_c) {
                min_c = edmonds.at<double>(i, j);
            }
        }
        t_to_s.push_back(min_c);
    }
    return t_to_s;
}

std::map<std::pair<int, int>, std::pair<int, int> > initialize_mates(cv::Mat & residual) {
    
    std::map<std::pair<int, int>, std::pair<int, int> > mates;
    std::pair<std::pair<int, int>, std::pair<int, int> > mate;
    for (int i = 0; i < residual.rows; i++) {
        for (int j = 0; j < residual.cols; j++) {
            mate.first = std::make_pair(i, j);
            mate.second = std::make_pair(-1, -1);
            mates.insert(mate);
        }
    }
    return mates;
}