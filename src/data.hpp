#pragma once

#include <string>
#include <vector>

constexpr int IMG_SIDE = 7;
constexpr int IMG_PIXELS = IMG_SIDE * IMG_SIDE;
constexpr int N_CLASSES = 3; // circle, square, triangle
constexpr int NOISE = -1;    // no output of its own, target is (0, 0, 0)

struct Sample {
    int label;                  // 0 circle, 1 square, 2 triangle, NOISE
    std::string name;           // header line, e.g. "circle fig2"
    std::vector<double> pixels; // IMG_PIXELS values, 0 or 1, row by row
};

std::vector<Sample> loadSamples(const std::string &path);
std::string labelName(int label);
void printSample(const Sample &s);
