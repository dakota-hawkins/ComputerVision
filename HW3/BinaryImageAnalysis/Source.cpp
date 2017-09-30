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
    open_full = cv::imread("../BinaryImages/open_fist-bw.png",
                           cv::IMREAD_GRAYSCALE);
    //cv::Mat open_full_binary = binarize_image(open_full);
    cv::Mat b_img = binarize_image(open_full);
    cv::Mat test = recursive_label(b_img);
    // cv::Mat labeled_full = sequential_label(open_full_binary);
    test.convertTo(test, CV_8UC1);
    // cv::Mat colored_full = color_labels(test);
    cv::namedWindow("open-bw-full", cv::WINDOW_AUTOSIZE);
    cv::imshow("open-bw-full", open_full);
    cv::namedWindow("colored_full", cv::WINDOW_AUTOSIZE);
    //cv::setMouseCallback("colored_full", mouse_callback, &colored_full);
    cv::imshow("colored_full", test);
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
    for (int i = 0; i < b_img.rows; i++) {
        for (int j = 0; j < b_img.cols; j++) {
            if (b_img.at<uchar>(i, j) == 1) {
                neg_b.at<schar>(i, j) = -1; 
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
    return neg_b;
}

std::vector<std::pair<int, int> > get_neighbors(cv::Mat& img, int y, int x) {
    // Repsect image boundaries. 
    int row_start = std::max(0, y - 1);
    int row_end = std::min(img.rows, y + 1);
    int col_start = std::max(0, x - 1);
    int col_end = std::min(img.cols, x + 1);
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

// Sequential labelling algorithm.
cv::Mat sequential_label(cv::Mat& b_img){
    // instantiate `b_img` sized grayscale image of zeros for component labels
    cv::Mat labeled = cv::Mat::zeros(b_img.rows, b_img.cols, CV_8UC1);
    
    // used to keep track of labels
    int count = 0;

    // vector of sets for equivalence classes between labels.
    std::vector<std::set<int> > set_vec;
    
    // Label connected regions    
    for (int row = 0; row < b_img.rows - 1; row++) {
    // int row = 0;
    // while (count < 50) {
        for (int col = 0; col < b_img.cols - 1; col++) {
            if (b_img.at<uchar>(row + 1, col + 1) != 0) {
                // Get lablels for neighboring pixels. 
                int diag = labeled.at<uchar>(row, col);
                int left = labeled.at<uchar>(row + 1, col);
                int up = labeled.at<uchar>(row, col + 1);

                // No connections, create new label.
                if (diag == 0 && left == 0 && up == 0) {               
                    count ++;
                    labeled.at<uchar>(row + 1, col + 1) = count;
                    int new_class[1] = {count};
                    add_class(set_vec, new_class, 1);
                    // std::cout << "Count: "  << count << std::endl;
                    // print_equivalence_classes(set_vec);
                }

                // check if pixel connected to upper diagnol
                if (diag != 0) {
                    labeled.at<uchar>(row + 1, col + 1) = diag;
                    continue;
                } else {
                    // check if left is labelled
                    if (left != 0) {
                        // check if up is labelled too
                        if (up != 0) {
                            // assign to left label if same
                            if (up == left) {
                                labeled.at<uchar>(row + 1, col + 1) = left;
                                continue;
                            } else {
                                // update labels, map pixel to minimum label
                                // map max label to minimum
                                int small = std::min(up, left);
                                int big = std::max(up, left);
                                labeled.at<uchar>(row + 1, col) = small;
                                labeled.at<uchar>(row, col + 1) = small;
                                labeled.at<uchar>(row + 1, col + 1) = small;
                                set_equivalencies(set_vec, small, big);
                                continue;
                            }
                        // connect to left if up in not labelled
                        } else {
                            labeled.at<uchar>(row + 1, col + 1) = left;
                            continue;
                        }
                    // check up label
                    } else if (up != 0) {
                        // connect pixel to up
                        labeled.at<uchar>(row + 1, col + 1) = up;
                        continue;
                    }
                }
            }
        }
        // row ++;
    }
    std::cout << "Count: " << count << std::endl;
    print_equivalence_classes(set_vec);
    
    // Map equivalent labels to each other. 
    // only do this if count > 0
    for (int row = 0; row < labeled.rows; row++) {
        for (int col = 0; col < labeled.cols; col++) {
            if (labeled.at<uchar>(row, col) != 0) {
                labeled.at<uchar>(row, col) = get_equivalency(set_vec, labeled.at<uchar>(row, col));
            }
        }
    }

    return labeled;
}
void add_class(std::vector<std::set<int> >& class_vec, int elements[], int n) {
    std::set<int> new_set(elements, elements + n);
    class_vec.push_back(new_set);
}

// Generate equivalence classes.
void set_equivalencies(std::vector<std::set<int> >& class_vec, int label1, int label2) {
    using ::std::cout;
    using ::std::endl;

    // cout << "label1: " << label1 << endl;
    // cout << "label2: " << label2 << endl;
    if (class_vec.size() == 0) {
        // cout << "Instantiating first set: {" << label1 << ", " << label2 << "}\n";
        int new_class[2] = {label1, label2};
        add_class(class_vec, new_class, 2);
    } else {
        int set1 = get_equivalency(class_vec, label1);
        int set2 = get_equivalency(class_vec, label2);
        // cout << "set1 index: " << set1 - 1 << endl;
        // cout << "set1: " << endl;
        // if (set1 - 1 > 0) {
        //     print_set(class_vec[set1 - 1]);
        // } else {
        //     cout << "{}\n";
        // }
        
        // cout << "set2 index: " << set2 - 1 << endl;
        // cout << "set2: " << endl;
        // if (set1 - 1 > 0) {
        //     print_set(class_vec[set2 - 1]);
        // } else {
        //     cout << "{}\n";
        // }

        if (set1 == 0 && set2 == 0) {
            // cout << "Neither label contained in class list\n";
            // cout << "Adding new set." << endl;
            int new_class[2] = {label1, label2};
            add_class(class_vec, new_class, 2);
            // print_equivalence_classes(class_vec);
        } else if (set1 != 0 && set2 == 0) {
            // cout << "Set2 not found in class list, adding to set 1 class." << endl;
            class_vec[set1 - 1].insert(label2);
            // print_equivalence_classes(class_vec);
        } else if (set1 == 0 && set2 != 0) {
            // cout << "Set1 not found in class list, adding to set 2 class." << endl;
            class_vec[set2 - 1].insert(label1);
            // print_equivalence_classes(class_vec);
        } else if (set1 != set2) {
            // cout << "Both labels already found in set list. Merging classes." << endl;
            merge_classes(class_vec, set1 - 1, set2 - 1);
            // print_equivalence_classes(class_vec);
        }
    }
}


void merge_classes(std::vector<std::set<int> >& class_vec,
    int set1_idx, int set2_idx) {
    // Implementation of removal borrowed from stack exchange.
    // https://stackoverflow.com/questions/39912/how-do-i-remove-an-item-from-a-stl-vector-with-a-certain-value
    
    std::set<int>::iterator it = class_vec[set2_idx].begin();
    std::set<int>::iterator end = class_vec[set2_idx].end();
    for (it; it != end; it++) {
        class_vec[set1_idx].insert(*it);
    }

    if (set2_idx != class_vec.size() - 1) {
        std::swap(class_vec[set2_idx], class_vec.back());
    }
    if (class_vec.size() > 1) {
        class_vec.pop_back();
    }
}


void print_equivalence_classes(std::vector<std::set<int> >& class_vec) {
    using ::std::cout;
    using ::std::endl;
    cout << "Number of sets: " << class_vec.size() << endl;
    for (int i = 0; i < class_vec.size(); i++) {
        print_set(class_vec[i]);
        cout << endl;
        // cout << "Set " << i + 1<< ":" << endl;
        // std::set<int>::iterator start = class_vec[i].begin();
        // std::set<int>::iterator end = class_vec[i].end();

        // cout << "{";
        // for (start; start != end; start++) {
        //     cout << *start << ", ";
        // }
        // cout << "}" << endl << endl;
    }
}

// Get equivalency class membership for element.
int get_equivalency(std::vector<std::set<int> >& class_vec, int label) {
    if (class_vec.size() != 0) {
        for (int i = 0; i < class_vec.size(); i++) {
            if (class_vec[i].find(label) != class_vec[i].end()) {
                return i + 1;
            }
        }
    }
    return 0;
}

// cv::Mat color_labels_random(cv::Mat& label_img) {
//     std::map<int, cv::Vec3b> color_map;
// }


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
