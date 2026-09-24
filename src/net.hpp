#pragma once

#include "data.hpp"

#include <string>
#include <vector>

constexpr double REJECT_THRESHOLD = 0.5; // max output below this -> noise

struct TrainResult {
    int epochs;
    bool converged; // every sample reached E <= eps
};

class Network {
  public:
    // sizes = {inputs, hidden..., outputs}, e.g. {49, 8, 3}
    Network(const std::vector<int> &sizes, unsigned seed);

    const std::vector<double> &forward(const std::vector<double> &input);
    double error(const std::vector<double> &target) const;
    void backward(const std::vector<double> &target, double alpha);

    TrainResult train(const std::vector<Sample> &set, double alpha,
                      double eps, int maxEpochs);
    int classify(const std::vector<double> &input); // class index or NOISE

  private:
    std::vector<int> n;                            // n[k]: neurons in layer k
    std::vector<std::vector<std::vector<double>>> w; // w[k][i][j]
    std::vector<std::vector<double>> y;            // y[k][i]
    std::vector<std::vector<double>> delta;        // delta[k][i]
};
