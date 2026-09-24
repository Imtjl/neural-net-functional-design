#include "net.hpp"

#include <algorithm>
#include <cmath>
#include <random>

static double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// One-hot target: (1,0,0) circle, (0,1,0) square, (0,0,1) triangle,
// (0,0,0) noise.
static std::vector<double> targetFor(int label) {
    std::vector<double> t(N_CLASSES, 0.0);
    if (label != NOISE)
        t[label] = 1.0;
    return t;
}

Network::Network(const std::vector<int> &sizes, unsigned seed)
    : n(sizes), w(sizes.size()), y(sizes.size()), delta(sizes.size()) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uniform(-0.5, 0.5);

    for (size_t k = 0; k < n.size(); ++k) {
        y[k].assign(n[k], 0.0);
        delta[k].assign(n[k], 0.0);
    }
    // Layer 0 is the input, it has no weights.
    for (size_t k = 1; k < n.size(); ++k) {
        w[k].assign(n[k], std::vector<double>(n[k - 1]));
        for (auto &row : w[k])
            for (double &v : row)
                v = uniform(rng);
    }
}

// y_i^k = f( sum_j y_j^(k-1) * w_ij^k ),  y^0 = input
const std::vector<double> &Network::forward(const std::vector<double> &input) {
    y[0] = input;
    for (size_t k = 1; k < n.size(); ++k)
        for (int i = 0; i < n[k]; ++i) {
            double sum = 0.0;
            for (int j = 0; j < n[k - 1]; ++j)
                sum += y[k - 1][j] * w[k][i][j];
            y[k][i] = sigmoid(sum);
        }
    return y.back();
}

// E = 1/2 * sum_i (t_i - y_i^N)^2, for the last forward() call
double Network::error(const std::vector<double> &target) const {
    double e = 0.0;
    for (int i = 0; i < n.back(); ++i) {
        double diff = target[i] - y.back()[i];
        e += diff * diff;
    }
    return 0.5 * e;
}

// Backpropagation for the last forward() call.
void Network::backward(const std::vector<double> &target, double alpha) {
    const size_t N = n.size() - 1;

    // Output layer: delta_i^N = y_i^N (1 - y_i^N) (t_i - y_i^N)
    for (int i = 0; i < n[N]; ++i)
        delta[N][i] = y[N][i] * (1.0 - y[N][i]) * (target[i] - y[N][i]);

    // Hidden layers: delta_i^k = y_i^k (1 - y_i^k) sum_j delta_j^(k+1) w_ji^(k+1)
    for (size_t k = N - 1; k >= 1; --k)
        for (int i = 0; i < n[k]; ++i) {
            double sum = 0.0;
            for (int j = 0; j < n[k + 1]; ++j)
                sum += delta[k + 1][j] * w[k + 1][j][i];
            delta[k][i] = y[k][i] * (1.0 - y[k][i]) * sum;
        }

    // All deltas above used the old weights, so update only now:
    // w_ij^k += alpha * delta_i^k * y_j^(k-1)
    for (size_t k = 1; k <= N; ++k)
        for (int i = 0; i < n[k]; ++i)
            for (int j = 0; j < n[k - 1]; ++j)
                w[k][i][j] += alpha * delta[k][i] * y[k - 1][j];
}

// One epoch = one pass over all samples. A sample whose error is above eps
// triggers a weight update. Training stops when a whole epoch passes with
// every sample at E <= eps.
TrainResult Network::train(const std::vector<Sample> &set, double alpha,
                           double eps, int maxEpochs) {
    for (int epoch = 1; epoch <= maxEpochs; ++epoch) {
        bool allLearned = true;
        for (const Sample &s : set) {
            std::vector<double> t = targetFor(s.label);
            forward(s.pixels);
            if (error(t) > eps) {
                backward(t, alpha);
                allLearned = false;
            }
        }
        if (allLearned)
            return {epoch, true};
    }
    return {maxEpochs, false};
}

// Index of the largest output, or NOISE if even that one is below the
// threshold (no output says "this is my figure").
int Network::classify(const std::vector<double> &input) {
    const std::vector<double> &out = forward(input);
    int best = (int)(std::max_element(out.begin(), out.end()) - out.begin());
    return out[best] < REJECT_THRESHOLD ? NOISE : best;
}
