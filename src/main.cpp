#include "data.hpp"
#include "net.hpp"

#include <cstdio>
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

// Trains one network and shows its answer on every test image.
static void runDemo(const std::vector<Sample> &train,
                    const std::vector<Sample> &test) {
    Network net({IMG_PIXELS, 8, N_CLASSES}, 1);
    TrainResult r = net.train(train, 0.3, 0.1, 10000);
    std::cout << "structure 49-8-3, converged: "
              << (r.converged ? "yes" : "no") << ", epochs: " << r.epochs
              << "\n\n";

    int correct = 0;
    for (const Sample &s : test) {
        const std::vector<double> &out = net.forward(s.pixels);
        std::printf("%.2f %.2f %.2f  ", out[0], out[1], out[2]);
        int predicted = net.classify(s.pixels);
        bool ok = predicted == s.label;
        correct += ok;
        std::cout << (ok ? "OK    " : "FAIL  ") << s.name << " -> "
                  << labelName(predicted) << "\n";
    }
    std::cout << "\ntest: " << correct << "/" << test.size() << " correct\n";
}

int main(int argc, char **argv) {
    std::string mode = argc > 1 ? argv[1] : "demo";

    try {
        std::vector<Sample> train = loadSamples("data/train.txt");
        std::vector<Sample> test = loadSamples("data/test.txt");
        printSummary("data/train.txt", train);
        printSummary("data/test.txt", test);

        if (mode == "demo") {
            runDemo(train, test);
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
