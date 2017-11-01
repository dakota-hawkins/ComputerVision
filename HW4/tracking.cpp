// Author: Dakota Hawkins
// Class: CS585 - HW4

// This program tracks the position of localized bats and fish in videos. 


#include "tracking.hpp"

using ::std::cout;
using ::std::cerr;
using ::std::endl;


int main() {

    // std::vector<std::string> files = get_file_names("./BatSegmentation/file_list.txt");
    // cv::Mat test_img = csv_to_img(files[0]);
    // cv::threshold(test_img, test_img, 0.5, 255, cv::THRESH_BINARY_INV);
    // cv::namedWindow("test");
    // cv::imshow("test", test_img);
    // cv::waitKey(0);
    // std::vector<std::string> sub_files(&files[0], &files[20]);
    // std::vector<cv::Mat> bats = file_list_to_data_list(sub_files, ".csv");
    
    // img_vec_to_file(bats, new_names);
    // cv::Mat data = bats[0];
    // cv::Mat colored_segments = color_labels(data);
    // cv::namedWindow("test", cv::WINDOW_NORMAL);
    // cv::resizeWindow("test", 512, 512);
    // cv::imshow("test", colored_segments);
    // cv::waitKey(0);
    // std::vector<cv::Point> p0, p1, vel;
    // p0.push_back(cv::Point(0, 1));
    // p0.push_back(cv::Point(2, 4));
    // p0.push_back(cv::Point(4, 5));
    // p1.push_back(cv::Point(1, 0));
    // p1.push_back(cv::Point(3, 3));
    // vel.push_back(cv::Point(0, 0));
    // vel.push_back(cv::Point(0, 0));
    // vel.push_back(cv::Point(0, 0));
    // p1.push_back(cv::Point2f(3, 6));
    // p1.push_back(cv::Point2f(7, 7));
    // std::vector<std::vector<double> > cost_m = cost_matrix(p0, p1, vel);
    // double row1[4] = {82, 83, 69, 92};
    // double row2[4] = {77, 37, 49, 92};
    // double row3[4] = {11, 69, 5, 86};
    // double row4[4] = {8, 9, 98, 23};
    // vector<double> v1(row1, row1 + 4);
    // vector<double> v2(row2, row2 + 4);
    // vector<double> v3(row3, row3 + 4);
    // vector<double> v4(row4, row4 + 4);
    // cost_m.push_back(v1);
    // cost_m.push_back(v2);
    // cost_m.push_back(v3);
    // cost_m.push_back(v4);


    // for (int i = 0; i < cost_m.size(); i ++) {
    //     for (int j = 0; j < cost_m[i].size(); j++) {
    //         cout << cost_m[i][j] << ", ";
    //     }
    //     cout << endl;
    // }

    // HungarianAlgorithm center_assignment;
    // std::vector<int> assignments;
    // center_assignment.Solve(cost_m, assignments);
    // cout << "Assignments: " << endl;
    // for (int i = 0; i < assignments.size(); i++) {
    //     cout << assignments[i] << endl;
    // }
    // cv::KalmanFilter kalman = initialize_kalman(p0[0]);
    analyze_bats();
    
    return 0;
}

void analyze_bats() {
    std::vector<std::string> center_files = get_file_names("./BatLocalization/file_list.txt");
    std::vector<std::string> csv_files = get_file_names("./BatSegmentation/file_list.txt");
    cv::namedWindow("t0", cv::WINDOW_NORMAL);
    cv::resizeWindow("t0", 700, 700);
    cv::namedWindow("t1", cv::WINDOW_NORMAL);
    cv::resizeWindow("t1", 700, 700);
    std::vector<cv::KalmanFilter> kalman_filters;
    std::vector<cv::Point> t0_centers, t1_centers;
    cv::Mat frame0, frame1;
    t0_centers = read_centers(center_files[0]);
    // std::vector<cv::Point> test = read_centers(center_files[0]);
    // t0_centers.push_back(test[0]);
    frame0 = csv_to_img(csv_files[0]);
    cv::threshold(frame0, frame0, 0.5, 255, cv::THRESH_BINARY_INV);
    cv::cvtColor(frame0, frame0, cv::COLOR_GRAY2BGR);

    // Initialize Kalman filters
    for (int i = 0; i < t0_centers.size(); i++) {
        kalman_filters.push_back(initialize_kalman(t0_centers[i]));
        kalman_loop(kalman_filters[i], t0_centers[i]);
    }

    // Center assignment variables
    HungarianAlgorithm center_assignment;
    std::vector<int> assignments;
    std::vector<std::vector<double> > edmonds;
    std::vector<cv::Point> velocities;

    // Trajectory tracking variables
    std::vector<std::vector<cv::Point> > trajectories;
    std::map<std::pair<int, int>, int> point_to_trajectory;


    // initialize trajectory objects
    init_trajectories(t0_centers, trajectories, point_to_trajectory);
    // initialize velocity vectors
    velocities = calculate_velocities(t0_centers, t0_centers);
    
    
    std::vector<cv::Point> kalman_centers;
    for (int i = 1; i < center_files.size(); i++) {
        std::vector<cv::Point> matched, unmatched_t0, unmatched_t1;
        std::set<int> t1_indices;
        std::vector<cv::Point> t1_centers = read_centers(center_files[i]);
        cout << "Past centers: " << t0_centers.size() << endl;
        cout << "Current centers: " << t1_centers.size() << endl;
        cout << "Number of trajectories saved: " << trajectories.size() << endl;
        cout << "Number of velocities saved: " << velocities.size() << endl;
        cv::Mat frame1 = csv_to_img(csv_files[i]);
        cv::threshold(frame1, frame1, 0.5, 255, cv::THRESH_BINARY_INV);
        cv::cvtColor(frame1, frame1, cv::COLOR_GRAY2BGR);
        
        // prune centers to those that are not closely related to other centers
        edmonds = cost_matrix(t0_centers, t1_centers, velocities);
        std::vector<cv::Point> old_filtered = filter_old_centers(edmonds, t0_centers, 70);
        std::vector<cv::Point> new_filtered = filter_new_centers(edmonds, t1_centers, 70);
        cout << "old filtered objects: " << old_filtered.size() << endl;
        cout << "new filtered objects: " << new_filtered.size() << endl;
        draw_centers(frame1, old_filtered, cv::Scalar(0, 0, 255));
        draw_centers(frame0, t0_centers, cv::Scalar(255, 0, 0));
        draw_centers(frame1, new_filtered, cv::Scalar(255, 0, 0));
        
        // old filtered centers likely left image => terminate trajectory
        terminate_trajectories(old_filtered, trajectories, point_to_trajectory);

        // new filtered object likely new objects => add new trajectory
        for (int j = 0; j < new_filtered.size(); j++) {
            add_trajectory(new_filtered[j], trajectories, point_to_trajectory);
        }

        // remove filtered centers from t0 and t1
        cout << "Filtering centers." << endl;
        remove_centers(t0_centers, old_filtered);
        remove_centers(t1_centers, new_filtered); // gonna want to keep t1_centers

        // get velocities for kept t0_centers
        cout << "Getting velocities for kept t0." << endl;
        velocities.clear();
        for (int j = 0; j < t0_centers.size(); j++) {
            velocities.push_back(cv::Point(0, 0));
        }
        // velocities = get_velocities(t0_centers, trajectories, point_to_trajectory);
        

        // recalculate distance matrix
        cout << "Recalculating cost matrix." << endl;
        edmonds = cost_matrix(t0_centers, t1_centers, velocities);

        // assign left over t0 centers to t1 centers
        center_assignment.Solve(edmonds, assignments);

        // unassigned t0 clusters likely due to occlusion
        // std::vector<cv::Point> unmatched_t0 = unmatched_t0_centers(assignments, t0_centers);
        cout << "Dealing with occlusion" << endl;
        unmatched_t0 = unmatched_t0_centers(assignments, t0_centers);
        if (unmatched_t0.size() > 0) {
            // likely occluded, assign to nearby center + <dx, dy>
            cv::Point closest, new_center;
            for (int j = 0; j < unmatched_t0.size(); j++) {
                // find past trajectory
                int traj_idx = find_trajectory(unmatched_t0[j], point_to_trajectory);
                // find nearest center
                closest = find_nearest_center(unmatched_t0[j], t1_centers);
                // update new center to estimated center using toward nearest
                // center
                new_center = (closest - unmatched_t0[j]) + unmatched_t0[j];
                new_center = ensure_unique_center(new_center, point_to_trajectory);
                // add estimated center to trajectories.
                trajectories[traj_idx].push_back(new_center);
            }
        }

        // unassigned t1 center likely due to unocclusion
        cout << "Dealing with innocculsion." << endl;
        unmatched_t1 = unmatched_t1_centers(assignments, t1_centers);
        if (unmatched_t1.size() > 0) {
            cv::Point closest, new_center;
            std::vector<cv::Point> new_traj;
            for (int j = 0; j < unmatched_t1.size(); j++) {
                // find closest t0 center, likely from there
                closest = find_nearest_center(unmatched_t1[j], t0_centers);
                new_center = (unmatched_t1[j] - closest) + closest;
                new_center = ensure_unique_center(new_center, point_to_trajectory);
                add_trajectory(new_center, trajectories, point_to_trajectory);
                update_trajectory(new_center, unmatched_t1[j], trajectories, point_to_trajectory);
            }
        }
        // matched t0 to t1 centers:
        cout << "Dealing with matched centers." << endl;
        for (int j = 0; j < assignments.size(); j++) {
            update_trajectory(t0_centers[j], t1_centers[assignments[j]], trajectories, point_to_trajectory);
        }

        // get current tracked points
        cout << "Getting current tracked points." << endl;
        t0_centers = get_current_points(trajectories);
        // get new velocities
        cout << "Updating velocities." << endl;
        velocities = get_velocities(t0_centers, trajectories, point_to_trajectory);

        cv::imshow("t0", frame0);
        cv::imshow("t1", frame1);
        cv::waitKey(0);
        cout << i << "/" << center_files.size() << endl;
        frame0 = frame1;
    }
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

std::vector<cv::Point> read_centers(const std::string& center_file) {
    std::ifstream read_file(center_file.c_str());
    std::vector<cv::Point> centers;
    std::string line;
    std::vector<std::string> split_vec;
    while (getline(read_file, line)) {
        boost::trim(line);
        boost::split(split_vec, line, boost::is_any_of(","));
        int i_value;
        cv::Point c_point;
        for (int i =0; i < 2; i++) {
            std::string s_value = split_vec[i];
            std::stringstream str_to_int(s_value);
            str_to_int >> i_value;
            if (i == 0) {
                c_point.x = i_value;
            } else {
                c_point.y = i_value;
            }
        }
        centers.push_back(c_point);
    }
    return centers;
}

void img_vec_to_file(std::vector<cv::Mat> images, std::vector<std::string> file_names) {
    for (int i = 0; i < images.size(); i++) {
        cv::imwrite(file_names[i], images[i]);
    }
}

std::vector<std::vector<double> > cost_matrix(std::vector<cv::Point> t0_centers, std::vector<cv::Point> t1_centers, std::vector<cv::Point> velocities) {
    std::vector<std::vector<double> > cost;
    for (int i = 0; i < t0_centers.size(); i++) {
        std::vector<double> new_row;
        for (int j = 0; j < t1_centers.size(); j++) {
            double edge = 1 + edge_length(t0_centers[i], t1_centers[j], velocities[i]);
            new_row.push_back(edge);
        }
        cost.push_back(new_row);
    }
    return cost;
}

double edge_length(cv::Point center1, cv::Point center2, cv::Point velocity) {
    // calculate distance between possible new center, and projected new center
    return cv::norm((center1 + velocity) - center2);
}

cv::KalmanFilter initialize_kalman(cv::Point first_measure) {
    cv::KalmanFilter kalman(4, 2, 0);
    // cv::Mat transition_matrix = cv::Mat::zeros(4, 4, CV_64F);
    cv::Mat_<float> transition_matrix = (cv::Mat_<float>(4, 4) << 1, 0, 1, 0,   0, 1, 0, 1,   0, 0, 1, 0,   0, 0, 0, 1);

    // Transition matrix init
    // x y vx vy
    // 1 0  1  0 x
    // 0 1  0  1 y
    // 0 0  1  0 vx
    // 0 0  0  1 vy
    kalman.transitionMatrix = transition_matrix;

    // init state
    kalman.statePre.at<float>(0) = first_measure.x;
    kalman.statePre.at<float>(1) = first_measure.y;
    kalman.statePre.at<float>(2) = 0;
    kalman.statePre.at<float>(3) = 0;

    cv::setIdentity(kalman.measurementMatrix);
    cv::setIdentity(kalman.processNoiseCov, cv::Scalar::all(1e-4));
    cv::setIdentity(kalman.measurementNoiseCov, cv::Scalar::all(1e-4));
    cv::setIdentity(kalman.errorCovPost, cv::Scalar::all(0.1));

    return kalman;
}

cv::Point kalman_loop(cv::KalmanFilter & kalman, cv::Point measurement) {
    // prediction step
    cv::Mat predict_mat = kalman.predict();
    cv::Point prediction(predict_mat.at<float>(0), predict_mat.at<float>(1));

    // measurement matrix init
    cv::Mat_<float> measure_mat(2, 1);
    measure_mat(0) = measurement.x;
    measure_mat(1) = measurement.y;

    // correction step
    cv::Mat estimate_mat = kalman.correct(measure_mat);
    cv::Point estimate(estimate_mat.at<float>(0), estimate_mat.at<float>(1));
    measure_mat.release();
    estimate_mat.release();
    return estimate;
}


void draw_centers(cv::Mat & img, std::vector<cv::Point> centers, cv::Scalar color) {
    for (int i = 0; i < centers.size(); i++) {
        cv::circle(img=img, centers[i], 3, color, 2);
    }
}


void init_trajectories(std::vector<cv::Point> t0_centers, std::vector<std::vector<cv::Point> >& trajectory, std::map<std::pair<int, int>, int>& traj_index) {
    std::pair<int, int> temp_pair;
    for (int i = 0; i < t0_centers.size(); i++) {
        std::vector<cv::Point> new_trajectory;
        new_trajectory.push_back(t0_centers[i]);
        temp_pair = std::make_pair(t0_centers[i].x, t0_centers[i].y);
        traj_index[temp_pair] = i;
        trajectory.push_back(new_trajectory);
    }
}

void add_trajectory(cv::Point new_center, std::vector<std::vector<cv::Point> >& trajectories, std::map<std::pair<int, int>, int>& pair_to_idx) {
    std::pair<int, int> new_pair = std::make_pair(new_center.x, new_center.y);
    std::map<std::pair<int, int>, int>::iterator it = pair_to_idx.find(new_pair);
    // ensure new_center is unique among current centers
    if (it != pair_to_idx.end()) {
        cout << "Error: center already matched with a trajectory" << endl;
        return;
    }
    // new trajectory vector
    std::vector<cv::Point> new_trajectory;
    // push center onto trajectory
    new_trajectory.push_back(new_center);
    // push center trajectory onto trajectories
    trajectories.push_back(new_trajectory);
    // point center to trajectory
    pair_to_idx[new_pair] = int(trajectories.size());
}

void update_trajectories(std::vector<cv::Point> t0_centers, std::vector<cv::Point> t1_centers, std::vector<int> assignments, std::vector<std::vector<cv::Point> >& trajectory, std::map<std::pair<int, int>, int>& traj_index) {

    std::pair<int, int> temp_pair;
    cv::Point current_point, new_center;
    int current_traj_index, center_idx;
    for (int i = 0; i < assignments.size(); i++) {
        // extract previous center, and trajectory index for center
        current_point = t0_centers[i];
        temp_pair = std::make_pair(current_point.x, current_point.y);
        current_traj_index = traj_index[temp_pair];
        // cout << "extracted successfully" << endl;
        // extract assigned center from frame t
        center_idx = assignments[i];
        new_center = t1_centers[center_idx];
        // push new center onto associated trajectory
        trajectory[current_traj_index].push_back(new_center);
        // re-assign traj_index(center) -> index mapping for updated center
        traj_index.erase(temp_pair);
        temp_pair = std::make_pair(new_center.x, new_center.y);
        traj_index[temp_pair] = current_traj_index;
    }
}

void terminate_trajectories(std::vector<cv::Point> centers,
                            std::vector<std::vector<cv::Point> >& trajectories,
                            std::map<std::pair<int, int>, int> & point_to_traj) {
    std::pair<int, int> temp_pair;
    int idx;
    for (int i = 0; i < centers.size(); i++) {
        temp_pair = std::make_pair(centers[i].x, centers[i].y);
        idx = point_to_traj[temp_pair];
        trajectories[idx].push_back(cv::Point(-1, -1));
    }
}

void draw_trajectories(cv::Mat &img, std::vector<std::vector<cv::Point> > trajectories) {
    // constant color vector
    std::vector<cv::Scalar> color_vec;  //
    // Scalar is B, G, R
    color_vec.push_back(cv::Scalar(66, 66, 244)); // red
    color_vec.push_back(cv::Scalar(244, 110, 66));  // blue
    color_vec.push_back(cv::Scalar(66, 203, 244)); // yellow
    color_vec.push_back(cv::Scalar(45, 140, 40));  // green
    color_vec.push_back(cv::Scalar(27, 103, 196));  // orange
    color_vec.push_back(cv::Scalar(196, 27, 128));  // purple
    for (int i = 0; i < trajectories.size(); i++) {
        std::vector<cv::Point> current_trajectory = trajectories[i];
        for (int j = current_trajectory.size() - 1; j > 0; j--) {
            // go to next trajectory if trajectory has been terminated
            if (current_trajectory[j] == cv::Point(-1, -1)) {
                break;
            }
            cv::line(img, current_trajectory[j], current_trajectory[j - 1], color_vec[(i + 1) % 6], 3);
        }
    }
}

std::vector<cv::Point> calculate_velocities(std::vector<cv::Point> t0_centers, std::vector<cv::Point> matched_centers) {
    std::vector<cv::Point> velocities;
    for (int i = 0; i < t0_centers.size(); i++) {
        velocities.push_back(matched_centers[i] - t0_centers[i]);
    }
    return velocities;
}

std::vector<cv::Point> get_velocities(std::vector<cv::Point> centers, std::vector<std::vector<cv::Point> > trajectories, std::map<std::pair<int, int>, int> idx_to_traj) {
    std::vector<cv::Point> velocities;
    std::vector<cv::Point> current_trajectory;
    int idx, center_idx;
    cout << "n traj" << trajectories.size() << endl;
    for (int i = 0; i < centers.size(); i++) {
        cout << "finding trajectory: " << endl;
        idx = find_trajectory(centers[i], idx_to_traj);
        if (idx == -1) {
            cout << "Error: cannot calculate velocity for non-logged center." << endl;
        }
        current_trajectory = trajectories[idx];
        std::vector<cv::Point> current_trajectory = trajectories[idx];
        cout << "traj index: " << idx << endl;
        center_idx = current_trajectory.size();
        cout << "center index: " << center_idx;
        // no velocity if first point
        if (center_idx == 0) {
            cout << "flagged\n";
            cv::Point temp = cv::Point(0, 0);
            cout << "Pushing back 0s\n";
            velocities.push_back(temp);
        } else {
            velocities.push_back(trajectories[idx][center_idx] - trajectories[idx][center_idx - 1]);
        }
    }
    return velocities;
}

std::set<int> create_index_set(std::vector<cv::Point> t1_centers) {
    std::set<int> index_set;
    for (int i = 0; i < t1_centers.size(); i++) {
        index_set.insert(i);
    }
    return index_set;
}

std::vector<cv::Point> filter_old_centers(std::vector<std::vector<double> > cost_m, std::vector<cv::Point> centers, double d_thresh) {
    vector<cv::Point> old_centers; 
    for (int i = 0; i < cost_m.size(); i++) {
        bool passed = false;
        for (int j = 0; j < cost_m.size(); j++) {
            if (cost_m[i][j] < d_thresh) {
                passed = true;
            }
        }
        if (!passed) {
            old_centers.push_back(centers[i]);
        }
    }
    return old_centers;
}

std::vector<cv::Point> filter_new_centers(std::vector<std::vector<double> > cost_m, std::vector<cv::Point> centers, double d_thresh) {
    std::vector<cv::Point> new_centers;
    for (int j = 0; j < cost_m[0].size(); j++) {
        bool passed = false;
        for (int i = 0; i < cost_m.size(); i++) {
            if (cost_m[i][j] < d_thresh) {
                passed = true;
            }
        }
        if (!passed) {
            new_centers.push_back(centers[j]);
        }
    }
    return new_centers;
}

void remove_centers(std::vector<cv::Point>& original, std::vector<cv::Point>& del_centers) {
    for (int i = 0; i < del_centers.size(); i++) {
        original.erase(std::remove(original.begin(), original.end(), del_centers[i]), original.begin());
    }
}


std::vector<cv::Point> unmatched_t0_centers(std::vector<int> assignments, std::vector<cv::Point> t0_centers) {
    std::vector<cv::Point> unmatched;
    for (int i = 0; i < assignments.size(); i++) {
        if (assignments[i] == -1) {
            unmatched.push_back(t0_centers[i]);
        }
    }
    return unmatched;
}

std::vector<cv::Point> unmatched_t1_centers(std::vector<int> assignments, std::vector<cv::Point> t1_centers) {
    std::set<int> t1_indices = create_index_set(t1_centers);
    std::vector<cv::Point> unmatched;
    for (int i = 0; i < assignments.size(); i++) {
        t1_indices.erase(assignments[i]);
    }
    for (std::set<int>::iterator it = t1_indices.begin(); it != t1_indices.end(); it++) {
        unmatched.push_back(t1_centers[*it]);
    }
    return unmatched;
}

int find_trajectory(cv::Point center, std::map<std::pair<int, int>, int>& idx_to_traj) {
    std::pair<int, int> temp_pair = std::make_pair(center.x, center.y);
    std::map<std::pair<int, int>, int>::iterator it;
    it = idx_to_traj.find(temp_pair);
    if (it == idx_to_traj.end()) {
        return -1;
    }
    return it -> second;
}


cv::Point find_nearest_center(cv::Point center, std::vector<cv::Point> centers) {
    double min_dist = 10e6;
    cv::Point nearest;
    for (int i = 0; i < centers.size(); i++) {
        if (cv::norm(centers[i] - center) < min_dist) {
            min_dist = cv::norm(centers[i] - center);
            nearest = centers[i];
        }
    }
    return nearest;
}

cv::Point ensure_unique_center(cv::Point center, std::map<std::pair<int, int>, int>& idx_to_traj) {
    cv::Point new_center = center;
    std::pair<int, int> new_pair = std::make_pair(new_center.x, new_center.y);
    std::map<std::pair<int, int>, int>::iterator it = idx_to_traj.find(new_pair);
    int count = 0;
    while (it != idx_to_traj.end()) {
        if (count % 2 == 0) {
            new_center.x += -1;
        } else {
            new_center.y += -1;
        }
    }
    return new_center;
}

void update_trajectory(cv::Point old_center, cv::Point new_center, std::vector<std::vector<cv::Point> >& trajectories, std::map<std::pair<int, int>, int>& idx_to_traj) {
    int idx = find_trajectory(old_center, idx_to_traj);
    std::pair<int, int> new_point;
    if (idx != -1) {
        int new_idx = find_trajectory(new_center, idx_to_traj);
        if (new_idx == -1) {
            trajectories[idx].push_back(new_center);
            new_point = std::make_pair(new_center.x, new_center.y);
            idx_to_traj[new_point] = idx;
        } else {
            cout << "Error: center already matched to another trajectory." << endl;
            return;
        }
    } else {
        cout << "Error: old center not found. Attempting to create  new trajectory." << endl;
        idx = find_trajectory(new_center, idx_to_traj);
        if (idx == - 1) {
            add_trajectory(new_center, trajectories, idx_to_traj);
        } else {
            cout << "Error: new center already belongs to a current trajectory." << endl;
        }
    }
}

std::vector<cv::Point> get_current_points(std::vector<std::vector<cv::Point> > trajectories) {
    std::vector<cv::Point> current_points;
    for (int i = 0; i < trajectories.size(); i++) {
        int n = trajectories.size() - 1;
        if (trajectories[i][n] != cv::Point(-1, -1)) {
            current_points.push_back(trajectories[i][n]);
        }
    }
}