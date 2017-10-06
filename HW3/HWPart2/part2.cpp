#include "part2.hpp"

int main() {
    using ::std::cout;
    using ::std::endl;
    using ::std::cerr;
    if( __cplusplus == 201103L ) std::cout << "C++11\n" ;
    else if( __cplusplus == 199711L ) std::cout << "C++98\n" ;
    else std::cout << "pre-standard C++\n";
    analyze_bats()
    return 0;
}

void analyze_bats() {
  using ::std::cout;
  using ::std::endl;
  using ::std::cerr;

  std::vector<std::string> bats = list_files("../BatImages/Gray/");
  cv::namedWindow("bats");
  cv::Mat frame;

  for (int i = 0; i < bats.size(); i++) {
      frame = cv::imread(bats[i], CV_8UC1);
      if (frame.empty()) {
        cerr << "Can't read image data: " << bats[i] << endl;
        continue;
      }
      cv::imshow("bats", frame);
      cv::waitKey(1);
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
