#include "part2.hpp"

int main() {
    using ::std::cout;
    using ::std::endl;
    using ::std::cerr;

    analyze_bats();
    return 0;
}

void analyze_bats() {
  using ::std::cout;
  using ::std::endl;
  using ::std::cerr;

  std::vector<std::string> bats = list_files("../BatImages/Gray/");
  cv::namedWindow("bats - source", cv::WINDOW_NORMAL);
  cv::resizeWindow("bats - source", 512, 512);
  cv::namedWindow("bats - filtered", cv::WINDOW_NORMAL);
  cv::resizeWindow("bats - filtered", 512, 512);
  cv::namedWindow("bats - colored", cv::WINDOW_NORMAL);
  cv::resizeWindow("bats - colored", 512, 512);
  cv::Mat frame;
  
  for (int i = 0; i < bats.size(); i++) {
      if (bats[i].substr(bats[i].find_last_of(".") + 1) != "ppm") {
          continue;
      }
      frame = cv::imread(bats[i], CV_8UC1);
      if (frame.empty()) {
          cerr << "Can't read image data: " << bats[i] << endl;
          return;
      }
      cout << "i = " << i << endl;
      cv::Mat mask_3 = cv::Mat::ones(cv::Size(3, 3), CV_8UC1);
      cv::Mat mask_5 = cv::Mat::ones(cv::Size(5, 5), CV_8UC1);
      cv::Mat blurred, binary_bats, dilated, eroded;
      cv::blur(frame, blurred, cv::Size(5, 5), cv::Point(-1, -1), 4);

      adaptive_threshold(blurred, binary_bats, 15, -5);
      cv::dilate(binary_bats, dilated, mask_3);
      erosion(dilated, eroded, mask_3);
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
      int count = 0;
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
                  count ++;
                  trace_color = cv::Vec3b(66, 66, 244);  // red
              }
              trace_binary(label_vec[i], fly_or_glide, trace_color);
              cout << "fullness: " << fullness << endl;
              cv::imshow("bats - colored", fly_or_glide);
              cv::waitKey(1);
          }
      }
      cv::waitKey(0);
      cout << "count = " << count;

      cv::Mat selected_bats = cv::Mat::zeros(frame.size(), CV_8UC1);
      vector_to_img(selected_bats, bat_vec);
  }
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



cv::Mat convert_to_color(cv::Mat& label_img) {
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
              std::cout << "r = " << row << " c = " << col << colored.size() << std::endl;
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

// trace a binary image onto another
void trace_binary(cv::Mat & src, cv::Mat & dst, int value) { 
  using ::std::cerr;
  using ::std::endl;
  if (src.empty()) {
    cerr << "Source image empty." << endl;
  }
  if (dst.empty()) {
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC1);
  }

  if (dst.size() != src.size()) {
    cerr << "Source and destination image dimensions do not match." << endl;
  }

  for (int row = 0; row < src.rows; row++) {
      for (int col = 0; col < src.cols; col++) {
        if (src.at<uchar>(row, col) != 0) {
          dst.at<uchar>(row, col) = value;
        }
    }
  }
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
