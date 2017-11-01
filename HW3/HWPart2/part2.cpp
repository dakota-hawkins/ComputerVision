#include "part2.hpp"
/*
Dakota Hawkins
BU CS585
Script to analyze `bats`, `piano` and `pedestriant` datasets.

All imports in header file. 
*/

int main() {
    using ::std::cout;
    using ::std::endl;
    using ::std::cerr;

    // analyze_bats();
    analyze_piano();
    // analyze_people();
    return 0;
}

void analyze_bats() {
  using ::std::cout;
  using ::std::endl;
  using ::std::cerr;

//   std::vector<std::string> bats = list_files("../BatImages/Gray/");
  cv::namedWindow("bats - source", cv::WINDOW_NORMAL);
  cv::resizeWindow("bats - source", 512, 512);
  cv::namedWindow("bats - filtered", cv::WINDOW_NORMAL);
  cv::resizeWindow("bats - filtered", 512, 512);
  cv::namedWindow("bats - colored", cv::WINDOW_NORMAL);
  cv::resizeWindow("bats - colored", 512, 512);
  std::ifstream in("../BatImages/Gray/file_list.txt");
  std::string line;
  std::vector<std::string> bats;
  cout << "hi" << endl;
  while (std::getline(in, line)) {
    bats.push_back("../BatImages/Gray/" + line);
  }
  cv::Mat frame;
  std::ofstream out_file;
  out_file.open("bats_counts.csv");
  out_file << "file,in,out" << endl;
  for (int k = 0; k < bats.size(); k++) {
      if (bats[k].substr(bats[k].find_last_of(".") + 1) != "ppm") {
          continue;
      }
      frame = cv::imread(bats[k], CV_8UC1);
      if (frame.empty()) {
          cerr << "Can't read image data: " << bats[k] << endl;
          return;
      }
      std::stringstream convert;
      convert << k;
      std::string k_string = convert.str();

      std::string base_name = "BatImages/bats_" + k_string + "_";
      std::string binary_file = base_name + "binary.jpg";
      std::string labeled_file = base_name + "labels.jpg";

      cout << "k = " << k << endl;
      cv::Mat mask_3 = cv::Mat::ones(cv::Size(3, 3), CV_8UC1);
      cv::Mat mask_5 = cv::Mat::ones(cv::Size(5, 5), CV_8UC1);
      cv::Mat blurred, binary_bats, dilated, eroded;
      cv::blur(frame, blurred, cv::Size(5, 5), cv::Point(-1, -1), 4);

      adaptive_threshold(blurred, binary_bats, 15, -5);
      cv::dilate(binary_bats, dilated, mask_3);
      erosion(dilated, eroded, mask_3);
      cv::imwrite(binary_file, dilated);
      cv::threshold(dilated, dilated, 5, 1, cv::THRESH_BINARY);

      cv::Mat labeled = cv::Mat::zeros(frame.size(), CV_16UC1);
      int n_labels = recursive_label(dilated, labeled);
      std::vector<cv::Mat> label_vec = label_img_to_vector(labeled, n_labels);
      std::vector<cv::Mat> bat_vec;
      cv::Mat borders = cv::Mat::zeros(frame.size(), CV_8UC1);

      cv::imshow("bats - source", frame);
      cv::imshow("bats - filtered", binary_bats);

      cv::waitKey(1);
      cv::Mat fly_or_glide = cv::Mat(frame.rows, frame.cols, CV_8UC3, cv::Vec3b(255, 255, 255));
      int count_out = 0;
      int count_in = 0;
      for (int i = 0; i < n_labels; i++) {
          std::map<std::string, double> stats = calculate_statistics(label_vec[i]);
          if (stats["area"] > 50) {
              bat_vec.push_back(label_vec[i]);
              double x_bar = stats["x_bar"];
              double y_bar = stats["y_bar"];
              double theta_rad = stats["theta"] * 180 / M_PI;

              cv::Mat theta_mat = cv::getRotationMatrix2D(cv::Point2f(x_bar, y_bar), theta_rad, 1);
              cv::Mat rotated = cv::Mat::zeros(label_vec[i].size(), CV_8UC1);
              cv::warpAffine(label_vec[i],rotated, theta_mat, label_vec[i].size());
              cv::Rect box = cv::boundingRect(rotated);

              double fullness = double(stats["area"]) / (box.width * box.height);
              cv::Vec3b trace_color = cv::Vec3b(244, 110, 66); // blue
              if (fullness > 0.53) {
                  count_out ++;
                  trace_color = cv::Vec3b(66, 66, 244);  // red
              } else {
                  count_in++;
              }
              trace_binary(label_vec[i], fly_or_glide, trace_color);
            //   cout << "fullness: " << fullness << endl;
              cv::waitKey(1);
          }
      }
      cv::imshow("bats - colored", fly_or_glide);
      cv::waitKey(1);
      cv::Mat selected_bats = cv::Mat::zeros(frame.size(), CV_8UC1);
      vector_to_img(selected_bats, bat_vec);
      cv::imwrite(labeled_file, fly_or_glide);
      out_file << bats[k] << "," << count_in << "," << count_out << endl;
  }
  out_file.close();
}


void analyze_piano() {
    using ::std::cout;
    using ::std::cerr;
    using ::std::endl;


    std::ifstream in("../PianoImages/file_names.txt");
    std::string line;
    std::vector<std::string> piano_imgs;
    while (std::getline(in, line)) {
        piano_imgs.push_back("../PianoImages/" + line);
    }


    std::vector<cv::Mat> src_vec, gray_vec, diff_vec;
    cout << "reading files..." << endl;
    int n_imgs = 0;
    for (int i = 0; i < piano_imgs.size(); i++) {
        if (piano_imgs[i].substr(piano_imgs[i].find_last_of(".") + 1) != "png") {
            continue;
        }
        // read image
        cv::Mat frame = cv::imread(piano_imgs[i], CV_LOAD_IMAGE_COLOR);
        if (frame.empty()) {
            cerr << "Can't read image data: " << piano_imgs[i] << endl;
            return;
        }
        src_vec.push_back(frame);

        // convert to grayscale
        cv::Mat gray_img;
        cv::cvtColor(frame, gray_img, CV_BGR2GRAY);
        gray_vec.push_back(gray_img);
        // calculate brightness difference
        cv::Mat diff_mat; 
        if (src_vec.size() == 1) {
          diff_mat = cv::Mat::zeros(frame.size(), CV_8UC1);
        } else {
            cv::absdiff(gray_img, gray_vec[src_vec.size() - 2], diff_mat);
        }
        if (! diff_mat.empty()) {
          diff_vec.push_back(diff_mat);
        }
    }

    cv::namedWindow("piano - source", cv::WINDOW_NORMAL);
    cv::resizeWindow("piano - source", 640, 400);
    cv::namedWindow("piano - gray", CV_WINDOW_NORMAL);
    cv::resizeWindow("piano - gray", 640, 400);
    cv::namedWindow("piano - difference", CV_WINDOW_NORMAL);
    cv::resizeWindow("piano - difference", 640, 400);
    cv::namedWindow("modified", CV_WINDOW_NORMAL);
    cv::resizeWindow("modified", 640, 400);
    
    // abs difference is abelian, so just set t0 to t1 - t0
    std::string dir = "HandImages/";
    diff_vec[0] = diff_vec[1];
    for (int i = 0; i < src_vec.size(); i++) {
      std::string i_string;
      std::stringstream convert;
      convert << i;
      i_string = convert.str();

      std::string base_name = "piano_" + i_string + "_";
      cout << i + 1 << "/" << src_vec.size() << endl;
      cv::imshow("piano - source", src_vec[i]);
      cv::imshow("piano - gray", gray_vec[i]);
      cv::imshow("piano - difference", diff_vec[i]);

      // Files to write
      std::string source_name = dir + base_name + "original.jpg";
      std::string diff_name = dir + base_name + "difference.jpg";
      std::string binary_name = dir + base_name + "binary.jpg";
      std::string dilate_name = dir + base_name + "dilate.jpg";
      std::string extracted_name = dir + base_name + "final.jpg";
      
      // write images
      cv::imwrite(source_name, src_vec[i]);
      cv::imwrite(diff_name, diff_vec[i]);

      // blur and threshold on brightness changes between frames. 
      cv::Mat b_img, blurred;
      cv::blur(diff_vec[i], blurred, cv::Size(5, 5), cv::Point(-1, -1), 4);
      double_threshold(blurred, b_img, 10, 30);
    
      // write binary image
      cv::imwrite(binary_name, b_img);
      blurred.release();

      // dilate image to get surrounding regions
      cv::Mat open_mask = cv::Mat::ones(cv::Size(20, 20), CV_8UC1);
      cv::dilate(b_img, b_img, open_mask);

      // write dilated image
      cv::imwrite(dilate_name, b_img);
      open_mask.release();

      // Find the largest segmented region -> likely piano lady
      cv::threshold(b_img, b_img, 5, 1, CV_THRESH_BINARY);
      cv::Mat labeled;
      int n_labels = recursive_label(b_img, labeled);
      int max_area = 0;
      int max_idx = 0;
      
      cv::Mat hands = b_img;
      if (n_labels > 1) {
        std::vector<cv::Mat> segments = label_img_to_vector(labeled, n_labels);
        for (int seg_idx = 0; seg_idx < segments.size(); seg_idx++) {
            cv::Moments moments = cv::moments(segments[seg_idx], true);
            if (moments.m00 > max_area) {
                max_area = moments.m00;
                max_idx = seg_idx;
            }
        }
        hands = segments[max_idx];
      }

      labeled.release();
    
      // extract area from original image
      cv::threshold(hands, hands, 0.5, 255, CV_THRESH_BINARY);
      cv::Mat out = extract_image(src_vec[i], hands);

      // write image
      cv::imwrite(extracted_name, out);
      cv::imshow("modified", out);
      cv::waitKey(0);
    }

}


void analyze_people() {
    using ::std::cerr;
    using ::std::cout;
    using ::std::endl;

    std::vector<std::string> people = list_files("../PeopleImages/");
    std::vector<cv::Mat> src_vec, gray_vec, diff_vec;
    cout << "reading files..." << endl;
    // std::vector<std::string> file_list = get_file_names("../PeopleImages/file_names.txt");
    std::ifstream in("../PeopleImages/file_names.txt");
    std::string line;
    std::vector<std::string> image_files;
    while (std::getline(in, line)) {
        image_files.push_back("../PeopleImages/" + line);
    }
    
    for (int i = 0; i < image_files.size(); i++) {
        if (image_files[i].substr(image_files[i].find_last_of(".") + 1) != "jpg") {
            continue;
        }
        // read image
        cv::Mat frame = cv::imread(image_files[i], CV_LOAD_IMAGE_COLOR);
        if (frame.empty()) {
            cerr << "Can't read image data: " << image_files[i] << endl;
            return;
        }
        src_vec.push_back(frame);

        // convert to grayscale
        cv::Mat gray_img;
        cv::cvtColor(frame, gray_img, CV_BGR2GRAY);
        gray_vec.push_back(gray_img);
    }

    for (int i = 0; i < gray_vec.size(); i++) {
        cv::Mat diff_mat;
        if (i == 0 ) {
            cv::absdiff(gray_vec[i], gray_vec[i + 1], diff_mat);
        } else if (i == gray_vec.size() - 1) {
            cv::absdiff(gray_vec[i], gray_vec[i - 1], diff_mat);
        } else {
            cv::Mat diff1, diff2;
            cv::absdiff(gray_vec[i], gray_vec[i - 1], diff1);
            cv::absdiff(gray_vec[i], gray_vec[i + 1], diff2);
            diff_mat = diff1;
            // average_img(diff1, diff2, diff_mat);
            diff1.release();
            diff2.release();
        }
        diff_vec.push_back(diff_mat);
        diff_mat.release();
    }

    cv::namedWindow("people", CV_WINDOW_NORMAL);
    cv::namedWindow("diff", CV_WINDOW_NORMAL);
    cv::namedWindow("binary", CV_WINDOW_NORMAL);
    cv::namedWindow("isolated", CV_WINDOW_NORMAL);
    std::ofstream write_file;
    write_file.open("pedestrian_guesses.csv");
    write_file << "file,guess" << endl;

    std::string dir = "PeopleImages/";
    for (int k = 0; k < src_vec.size(); k++) {

        std::string k_string;
        std::stringstream convert;
        convert << k;
        k_string = convert.str();
  
        std::string base_name = dir + "people_" + k_string + "_";
        std::string binary_file = base_name + "binary.jpg";
        std::string diff_file = base_name + "difference.jpg";
        std::string labeled_file = base_name + "labels.jpg";

        // cv::imshow("people", src_vec[k]);
        // cv::imshow("diff", diff_vec[k]);
        cv::Mat blurred, b_img;
        // double_threshold(diff_vec[i], b_img, 20, 70);
        cv::blur(diff_vec[k], blurred, cv::Size(10, 10), cv::Point(-1, -1), 4);
        // double_threshold(blurred, b_img, 20, 40);
        // cv::threshold(blurred, b_img, 20, 255, CV_THRESH_BINARY);
        adaptive_threshold(blurred, b_img, 15, -6);
        blurred.release();
        cv::Mat labeled;
        cv::threshold(b_img, b_img, 5, 1, CV_THRESH_BINARY);
        int n_labels = recursive_label(b_img, labeled);
        std::vector<cv::Mat> label_vec = label_img_to_vector(labeled, n_labels);
        labeled.release();

        std::vector<double> y_bars;
        std::vector<double> x_bars;
        std::vector<double> areas;
        for (int i = 0; i < n_labels; i++) {
            std::map<std::string, double> stats = calculate_statistics(label_vec[i]);
            y_bars.push_back(stats["y_bar"]);
            x_bars.push_back(stats["x_bar"]);
            areas.push_back(stats["area"]);
        }

        std::set<int> matched;
        std::set<int> rep_idx;
        cout << "#: " << n_labels << endl;
        for (int i = 0; i < n_labels; i++) {
            std::set<int>::iterator it;
            // for (it = matched.begin(); it != matched.end(); it++) {
            //     cout << *it <<  ", ";
            // }
            // cout << endl;
            it = matched.find(i);
            if (it == matched.end() && areas[i] > 100) {
                matched.insert(i);
                rep_idx.insert(i);
                double y_bar = y_bars[i];
                double x_bar = x_bars[i];
                // cout << i << " (" << y_bar << ", " << x_bar << ") matched with:" << endl; 
                double y_thresh = 80;
                double x_thresh = 30;
                for (int j = 0; j < y_bars.size(); j++) {
                    if (j != i) {
                        bool in_y = (y_bar - y_thresh <= y_bars[j] && y_bar + y_thresh >= y_bars[j]);
                        bool in_x = (x_bar - x_thresh <= x_bars[j] && x_bar + x_thresh >= x_bars[j]);
                        if (in_y && in_x) {
                            matched.insert(j);
                            label_vec[i] += label_vec[j];
                        }
                    } 
                }
            }
        }

        std::set<int>::iterator it;
        std::vector<cv::Mat> combined_vec;
        for (int i = 0; i < label_vec.size(); i++) {
            it = rep_idx.find(i);
            if (it == rep_idx.end()) {
                label_vec[i].release();
            } else {
                combined_vec.push_back(label_vec[i]);
            }
        }
        write_file << image_files[k] << "," << combined_vec.size() << endl;
        cv::Mat combined_labels = cv::Mat(diff_vec[0].size(), CV_8UC1);
        vector_to_img(combined_labels, combined_vec);
        cv::Mat colored_labeled = color_labels(combined_labels);
        cv::Rect test_rect = cv::Rect(0, 0, 30, 80);
        cv::rectangle(colored_labeled, test_rect, cv::Scalar(100, 100, 100));
        cv::threshold(b_img, b_img, 0, 255, CV_THRESH_BINARY);
        cv::imshow("people", src_vec[k]);
        cv::imshow("diff", diff_vec[k]);
        cv::imshow("binary", b_img);
        cv::imshow("isolated", colored_labeled);
        cv::waitKey(1);
        cv::imwrite(binary_file, b_img);
        cv::imwrite(diff_file, diff_vec[k]);
        cv::imwrite(labeled_file, colored_labeled);
        b_img.release();
        labeled.release();
        combined_labels.release();
        colored_labeled.release();
    }
    write_file.close();
}


std::vector<std::string> list_files(std::string read_dir){
    DIR           *dirp;
    struct dirent *directory;
    const char * c_dir = read_dir.c_str();
    std::vector<std::string> files;
    dirp = opendir(c_dir);
    if (dirp) {
        while ((directory = readdir(dirp)) != NULL)
        {
          std::string file_name = (directory->d_name);
          files.push_back(read_dir + "/" + file_name);
        }

        closedir(dirp);
    }

    return files;
}


// trace a binary image onto another.
void trace_binary(cv::Mat & src, cv::Mat & dst, cv::Vec3b value) {
  using ::std::cerr;
  using ::std::endl;

  if (src.empty()) {
      cerr << "Source image empty." << endl;
  }
  if (dst.empty()) {
      cv::Mat dst = cv::Mat(src.size(), CV_8UC3, cv::Scalar(255, 255, 255));
  }

  if (dst.size() != src.size()) {
      cerr << "Source and destination image dimensions do not match." << endl;
  }

  for (int row = 0; row < src.rows; row++) {
      for (int col = 0; col < src.cols; col++) {
        if (src.at<uchar>(row, col) != 0) {
          dst.at<cv::Vec3b>(row, col) = value;
        }
      }
  }
}


cv::Mat extract_image(cv::Mat & src, cv::Mat & mask) {
  using ::std::cerr;
  using ::std::cout;
  using ::std::endl;

  if (src.empty()) {
      cerr << "Error: source image is empty." << endl;
  }

  if (src.size() != mask.size()) {
      cerr << "Error: mask and source image must have the same dimensions.";
  }
  int channels = src.channels();
  cv::Mat dst;
  // Source is gray image. 
  if (channels == 1) {
      dst = cv::Mat::zeros(src.size(), CV_8UC1);
  // Source is BGR
  } else if (channels == 3) {
      dst = cv::Mat(src.size(), src.type(), cv::Scalar(255, 255, 255));
  } else {
      cerr << "Error: unsupported image type with " << channels << " channels" << endl;
  }

  for (int row = 0; row < src.rows; row++) {
      for(int col = 0; col < src.cols; col++) {
        if (mask.at<uchar>(row, col) != 0) {
            if (channels == 1) {
                dst.at<uchar>(row, col) = src.at<uchar>(row, col);
            } else if (channels == 3) {
                dst.at<cv::Vec3b>(row, col) = src.at<cv::Vec3b>(row, col);
            } else {
                cerr << "Error: unsupported image type with " << channels << " channels" << endl;
            }
        }
      }
  }
  return dst;
}

void skin_detect(cv::Mat& src, cv::Mat& dst) {
	//Surveys of skin color modeling and detection techniques:
	//Vezhnevets, Vladimir, Vassili Sazonov, and Alla Andreeva. "A survey on pixel-based skin color detection techniques." Proc. Graphicon. Vol. 3. 2003.
	//Kakumanu, Praveen, Sokratis Makrogiannis, and Nikolaos Bourbakis. "A survey of skin-color modeling and detection methods." Pattern recognition 40.3 (2007): 1106-1122.

    // Implement Peer method
    if (dst.empty()) {
        dst = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);
    }
	for (int row = 0; row < src.rows; row++) {
		for (int col = 0; col < src.cols; col++) {
        bool skin_color = peer_metric(src.at<cv::Vec3b>(row, col));
        if (skin_color) {
            dst.at<uchar>(row, col) = 255;
        } else {
            dst.at<uchar>(row, col) = 0;
        }
		}
  }
}


// Function that detects whether a pixel belongs to skin based on RGB values.
bool peer_metric(cv::Vec3b& bgr) {
  int blue, green, red;
  blue = bgr[0];
  green = bgr[1];
  red = bgr[2];
  if (red > 95 && green > 40 && blue > 20) {
      if (my_max(bgr[0], bgr[1], bgr[2]) -  my_min(bgr[0], bgr[1], bgr[2]) > 15) {
        if (abs(bgr[2] - bgr[1]) > 15 && bgr[2] > bgr[1] && bgr[2] > bgr[0]) {
          return true;
        }
      }
  }
  return false;
}


// find the max of three integers
int my_max(int a, int b, int c) {
	int m = a;
	(void)((m < b) && (m = b));
	(void)((m < c) && (m = c));
	return m;
}


// find the minimum of three integers
int my_min(int a, int b, int c) {
	int m = a;
	(void)((m > b) && (m = b));
	(void)((m > c) && (m = c));
	return m;
}

void average_img(cv::Mat & img1, cv::Mat & img2, cv::Mat & dst) {
    using ::std::cerr;
    using ::std::cout;
    using ::std::endl;

    if (img1.empty()) {
        cerr << "Error: first image passed was empty." << endl;
        return;
    }

    if (img2.empty()) {
        cerr << "Error: second image passed was emtpy." << endl;
        return;
    }

    if (img1.channels() != 1) {
        cerr << "Error: expected single-channel image." << endl;
        return;
    }

    if (img2.channels() != 1) {
        cerr << "Error: expected single-channel image." << endl;
        return;
    }

    if (img1.type() != img2.type()) {
        cerr << "Error: expected image types to match" << endl;
        return;
    }
    
    if (img1.size() != img2.size()) {
        cerr << "Error: expected equal-sized images." << endl;
        return;
    }

    if (dst.empty()) {
        dst = cv::Mat::zeros(img1.size(), img1.type());
    }

    if (dst.size() != img1.size()) {
        cerr << "Error: expected output and input images to be equal dimensions." << endl;
        return;
    }

    for (int row = 0; row < img1.rows; row++) {
        for (int col = 0; col < img1.cols; col++) {
            int value = (img1.at<uchar>(row, col) + img2.at<uchar>(row, col));
            dst.at<uchar>(row, col) = value*0.5;
        }
    }
}