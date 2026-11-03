#include <algorithm>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>

#include "NAM.h"
#include "getImageList.h"

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "usage: run_dataset_linux INPUT_DIR OUTPUT_DIR\n";
        return 2;
    }
    const std::string input = std::string(argv[1]) + "/";
    const std::string output = std::string(argv[2]) + "/";
    mkdir(argv[2], 0755);

    std::unordered_map<std::string, double> param;
    param["mode"] = 1;
    param["saveHir"] = 60;
    param["lab"] = 0;
    param["homogeneous"] = 0;
    param["k"] = 2.0;
    param["shape"] = 0;
    param["ratio"] = 1.0;
    param["u"] = 1.9;
    param["var"] = 3.0;
    param["alpha"] = 1.0;
    param["beta"] = 1.97;
    param["gamma"] = 1.97;
    param["percent"] = 6.7;
    param["si"] = 67.0;
    param["size"] = 4700.0;
    param["change"] = 197;

    std::vector<std::string> images;
    getImageList(input.c_str(), images);
    std::sort(images.begin(), images.end());
    for (size_t index = 0; index < images.size(); ++index) {
        const std::string &name = images[index];
        std::vector<cv::Mat> maps, results;
        NAM(param, input + name, maps, results);
        cv::Mat image = cv::imread(input + name);
        if (image.empty()) {
            std::cerr << "Could not process " << name << "\n";
            return 1;
        }

        cv::Mat mark = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
        if (!maps.empty()) {
            const cv::Mat &segments = maps[0];
            for (int row = 0; row < image.rows; ++row) {
                for (int col = 0; col < image.cols; ++col) {
                    const bool vertical = row > 0 && row + 1 < image.rows &&
                        segments.at<float>(row - 1, col) != segments.at<float>(row + 1, col) &&
                        !mark.at<unsigned char>(row - 1, col) && !mark.at<unsigned char>(row + 1, col);
                    const bool horizontal = col > 0 && col + 1 < image.cols &&
                        segments.at<float>(row, col - 1) != segments.at<float>(row, col + 1) &&
                        !mark.at<unsigned char>(row, col - 1) && !mark.at<unsigned char>(row, col + 1);
                    if (!mark.at<unsigned char>(row, col) && (vertical || horizontal)) {
                        mark.at<unsigned char>(row, col) = 255;
                    }
                }
            }
        }
        const std::string stem = name.substr(0, name.find_last_of('.'));
        const std::string path = output + stem + "_Hier_" + std::to_string(results.size()) + "_Mark.png";
        if (!cv::imwrite(path, mark)) {
            std::cerr << "Could not write " << path << "\n";
            return 1;
        }
        std::cout << index + 1 << "/" << images.size() << " " << name << std::endl;
    }
    return 0;
}
