#include "data.hpp"
#include "net.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Training parameters from the task: alpha in 0.1..0.3, eps = 0.1.
// compare accepts another eps to show what longer training does.
constexpr double ALPHA = 0.3;
constexpr double EPS = 0.1;
constexpr int MAX_EPOCHS = 10000;
constexpr int SEEDS = 10; // compare: runs per structure

using Clock = std::chrono::steady_clock;

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

// "49-8-3" -> {49, 8, 3}
static std::vector<int> parseStructure(const std::string &text) {
    std::vector<int> sizes;
    std::istringstream in(text);
    std::string part;
    while (std::getline(in, part, '-'))
        sizes.push_back(std::stoi(part));
    if (sizes.size() < 3 || sizes.front() != IMG_PIXELS ||
        sizes.back() != N_CLASSES)
        throw std::runtime_error("structure must be 49-<hidden...>-3, got " +
                                 text);
    return sizes;
}

static std::string structureName(const std::vector<int> &sizes) {
    std::string s;
    for (size_t k = 0; k < sizes.size(); ++k)
        s += (k ? "-" : "") + std::to_string(sizes[k]);
    return s;
}

static int weightCount(const std::vector<int> &sizes) {
    int c = 0;
    for (size_t k = 1; k < sizes.size(); ++k)
        c += sizes[k] * sizes[k - 1];
    return c;
}

static int countCorrect(Network &net, const std::vector<Sample> &set) {
    int correct = 0;
    for (const Sample &s : set)
        correct += net.classify(s.pixels) == s.label;
    return correct;
}

// Memory for the numbers the algorithm keeps, double = 8 bytes.
// Inference: weights + output y of every neuron (y^0 is the input image).
// Training: inference + delta of every non-input neuron + target vector.
static void printMemory(const std::vector<int> &sizes, size_t trainImages) {
    int weights = weightCount(sizes);
    int outputs = 0, deltas = 0;
    for (size_t k = 0; k < sizes.size(); ++k) {
        outputs += sizes[k];
        if (k > 0)
            deltas += sizes[k];
    }
    size_t d = sizeof(double);
    size_t inference = (weights + outputs) * d;
    size_t training = inference + (deltas + N_CLASSES) * d;
    std::printf("memory, %zu B per number:\n", d);
    std::printf("  inference:    %5zu B = %d weights + %d neuron outputs\n",
                inference, weights, outputs);
    std::printf("  training:     %5zu B = inference + %d deltas + %d targets\n",
                training, deltas, N_CLASSES);
    std::printf("  training set: %5zu B = %zu images x %d pixels\n",
                trainImages * IMG_PIXELS * d, trainImages, IMG_PIXELS);
}

// Trains one network, measures it (task items 3.1-3.4) and shows its answer
// on every test image.
static void runDemo(const std::vector<Sample> &train,
                    const std::vector<Sample> &test,
                    const std::vector<int> &sizes) {
    Network net(sizes, 1);
    auto t0 = Clock::now();
    TrainResult r = net.train(train, ALPHA, EPS, MAX_EPOCHS);
    double trainMs =
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

    // One forward pass takes well under a microsecond, so time many of them
    // and divide. The sum keeps the compiler from dropping unused results.
    constexpr int FORWARD_RUNS = 100000;
    double sink = 0.0;
    auto t1 = Clock::now();
    for (int i = 0; i < FORWARD_RUNS; ++i)
        sink += net.forward(test[i % test.size()].pixels)[0];
    double forwardUs =
        std::chrono::duration<double, std::micro>(Clock::now() - t1).count() /
        FORWARD_RUNS;

    std::printf("structure %s (%d weights), seed 1\n",
                structureName(sizes).c_str(), weightCount(sizes));
    std::printf("training: converged %s, %d epochs, %.2f ms\n",
                r.converged ? "yes" : "no", r.epochs, trainMs);
    std::printf("forward pass: %.3f us (mean of %d runs, sink %.0f)\n",
                forwardUs, FORWARD_RUNS, sink);
    printMemory(sizes, train.size());
    std::cout << "\n";

    for (const Sample &s : test) {
        const std::vector<double> &out = net.forward(s.pixels);
        std::printf("%.2f %.2f %.2f  ", out[0], out[1], out[2]);
        int predicted = net.classify(s.pixels);
        std::cout << (predicted == s.label ? "OK    " : "FAIL  ") << s.name
                  << " -> " << labelName(predicted) << "\n";
    }
    std::cout << "\ntest: " << countCorrect(net, test) << "/" << test.size()
              << " correct\n";
}

// Averages over SEEDS trainings of one structure with one alpha.
struct RunStats {
    int converged = 0;
    double epochs = 0, trainMs = 0, trainOk = 0, testOk = 0;
    int testMin = 1 << 30, testMax = 0;
};

static RunStats evaluate(const std::vector<int> &sizes, double alpha,
                         double eps, const std::vector<Sample> &train,
                         const std::vector<Sample> &test) {
    RunStats st;
    for (int seed = 1; seed <= SEEDS; ++seed) {
        Network net(sizes, seed);
        auto t0 = Clock::now();
        TrainResult r = net.train(train, alpha, eps, MAX_EPOCHS);
        st.trainMs +=
            std::chrono::duration<double, std::milli>(Clock::now() - t0)
                .count();
        st.converged += r.converged;
        st.epochs += r.epochs;
        st.trainOk += countCorrect(net, train);
        int t = countCorrect(net, test);
        st.testOk += t;
        st.testMin = std::min(st.testMin, t);
        st.testMax = std::max(st.testMax, t);
    }
    st.epochs /= SEEDS;
    st.trainMs /= SEEDS;
    st.trainOk /= SEEDS;
    st.testOk /= SEEDS;
    return st;
}

static void printHeader(const char *firstColumn) {
    std::printf("%-12s %7s %9s %8s %9s %9s %10s\n", firstColumn, "weights",
                "converged", "epochs", "train_ms", "train_ok", "test_ok");
}

static void printRow(const std::string &label, int weights,
                     const RunStats &st, size_t trainSize, size_t testSize) {
    std::printf("%-12s %7d %6d/%-2d %8.1f %9.2f %6.1f/%-2zu %4.1f/%zu (%d..%d)\n",
                label.c_str(), weights, st.converged, SEEDS, st.epochs,
                st.trainMs, st.trainOk, trainSize, st.testOk, testSize,
                st.testMin, st.testMax);
}

// Trains every structure with SEEDS different initial weights and prints
// averages. The best structure is the one with the highest mean test
// accuracy (generalization); on a tie the one with fewer weights wins.
static void runCompare(const std::vector<Sample> &train,
                       const std::vector<Sample> &test, double eps) {
    const std::vector<std::vector<int>> structures = {
        {49, 4, 3},  {49, 8, 3},     {49, 16, 3},
        {49, 32, 3}, {49, 8, 8, 3},  {49, 16, 8, 3},
    };

    std::cout << "alpha " << ALPHA << ", eps " << eps << ", " << SEEDS
              << " seeds per structure\n\n";
    printHeader("structure");

    std::vector<int> best;
    double bestTest = -1.0;
    for (const auto &sizes : structures) {
        RunStats st = evaluate(sizes, ALPHA, eps, train, test);
        printRow(structureName(sizes), weightCount(sizes), st, train.size(),
                 test.size());
        if (st.testOk > bestTest ||
            (st.testOk == bestTest && weightCount(sizes) < weightCount(best))) {
            bestTest = st.testOk;
            best = sizes;
        }
    }
    std::cout << "\nbest: " << structureName(best) << "\n";
}

// The task says alpha is chosen empirically in 0..1 (0.1..0.3 recommended
// against overfitting). Trains one structure with each alpha.
static void runAlpha(const std::vector<Sample> &train,
                     const std::vector<Sample> &test,
                     const std::vector<int> &sizes) {
    const std::vector<double> alphas = {0.01, 0.05, 0.1, 0.2,
                                        0.3,  0.5,  0.8, 1.0};

    std::cout << "structure " << structureName(sizes) << ", eps " << EPS
              << ", " << SEEDS << " seeds per alpha\n\n";
    printHeader("alpha");
    for (double alpha : alphas) {
        RunStats st = evaluate(sizes, alpha, EPS, train, test);
        char label[16];
        std::snprintf(label, sizeof label, "%.2f", alpha);
        printRow(label, weightCount(sizes), st, train.size(), test.size());
    }
}

int main(int argc, char **argv) {
    std::string mode = argc > 1 ? argv[1] : "demo";

    try {
        std::vector<Sample> train = loadSamples("data/train.txt");
        std::vector<Sample> test = loadSamples("data/test.txt");
        printSummary("data/train.txt", train);
        printSummary("data/test.txt", test);
        std::cout << "\n";

        if (mode == "demo") {
            std::string structure = argc > 2 ? argv[2] : "49-8-3";
            runDemo(train, test, parseStructure(structure));
        } else if (mode == "alpha") {
            std::string structure = argc > 2 ? argv[2] : "49-8-3";
            runAlpha(train, test, parseStructure(structure));
        } else if (mode == "compare") {
            double eps = argc > 2 ? std::stod(argv[2]) : EPS;
            runCompare(train, test, eps);
        } else {
            std::cerr << "usage: nn demo [49-<hidden...>-3] | nn compare [eps] | "
                         "nn alpha [49-<hidden...>-3]\n";
            return 1;
        }
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
