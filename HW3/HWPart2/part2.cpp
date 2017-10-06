#include "part2.hpp"

int main() {
    using ::std::cout;
    using ::std::endl;
    cout << "hi! u suk." << endl;
    if( __cplusplus == 201103L ) std::cout << "C++11\n" ;
    else if( __cplusplus == 199711L ) std::cout << "C++98\n" ;
    else std::cout << "pre-standard C++\n" ;
    std::vector<std::string> bats = print_dir("../BatImages/Gray/");
    for (int i = 0; i < bats.size(); i++) {
        cout << bats[i] << endl;
    }
    return 0;
}

std::vector<std::string> print_dir(std::string read_dir){
    DIR           *dirp;
    struct dirent *directory;
    const char * c_dir = read_dir.c_str();
    std::vector<std::string> files;

    dirp = opendir(c_dir);
    if (dirp) {
        while ((directory = readdir(dirp)) != NULL)
        {
          files.push_back(directory->d_name);
        }

        closedir(dirp);
    }

    return files;
}