#include "data.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void printSummary(const std::string &path,
                         const std::vector<Sample> &set) {
    int count[N_CLASSES + 1] = {}; // last slot counts noise
    for (const Sample &s : set)
        ++count[s.label == NOISE ? N_CLASSES : s.label];

    std::cout << path << ": " << set.size() << " samples (";
    for (int i = 0; i < N_CLASSES; ++i)
        std::cout << labelName(i) << " " << count[i] << ", ";
    std::cout << "noise " << count[N_CLASSES] << ")\n";
}

int main(int argc, char **argv) {
    std::string mode = argc > 1 ? argv[1] : "demo";

    try {
        std::vector<Sample> train = loadSamples("data/train.txt");
        std::vector<Sample> test = loadSamples("data/test.txt");
        printSummary("data/train.txt", train);
        printSummary("data/test.txt", test);

        if (mode == "demo") {
            std::cout << "mode: demo (stub)\n";
        } else if (mode == "compare") {
            std::cout << "mode: compare (stub)\n";
        } else {
            std::cerr << "usage: nn [demo|compare]\n";
            return 1;
        }
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
