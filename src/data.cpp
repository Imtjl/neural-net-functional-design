#include "data.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

static const char *CLASS_NAMES[N_CLASSES] = {"circle", "square", "triangle"};

std::string labelName(int label) {
    return label == NOISE ? "noise" : CLASS_NAMES[label];
}

// Returns false for an unknown word.
static bool parseLabel(const std::string &word, int &label) {
    if (word == "noise") {
        label = NOISE;
        return true;
    }
    for (int i = 0; i < N_CLASSES; ++i)
        if (word == CLASS_NAMES[i]) {
            label = i;
            return true;
        }
    return false;
}

// File format: a header line "<label> <name>", then IMG_SIDE rows of
// '#' (black, 1) and '.' (white, 0). Blank lines between images.
std::vector<Sample> loadSamples(const std::string &path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("cannot open " + path);

    std::vector<Sample> samples;
    std::string line;
    int lineNo = 0;
    auto fail = [&](const std::string &msg) {
        throw std::runtime_error(path + ":" + std::to_string(lineNo) + ": " +
                                 msg);
    };

    while (std::getline(in, line)) {
        ++lineNo;
        if (line.empty())
            continue;

        Sample s;
        s.name = line;
        std::string word;
        std::istringstream(line) >> word;
        if (!parseLabel(word, s.label))
            fail("unknown label '" + word + "'");

        for (int r = 0; r < IMG_SIDE; ++r) {
            if (!std::getline(in, line))
                fail("image is cut off, expected " +
                     std::to_string(IMG_SIDE) + " rows");
            ++lineNo;
            if ((int)line.size() != IMG_SIDE)
                fail("row must have " + std::to_string(IMG_SIDE) + " chars");
            for (char c : line) {
                if (c != '#' && c != '.')
                    fail("only '#' and '.' are allowed");
                s.pixels.push_back(c == '#' ? 1.0 : 0.0);
            }
        }
        samples.push_back(s);
    }
    return samples;
}

void printSample(const Sample &s) {
    std::cout << s.name << "\n";
    for (int r = 0; r < IMG_SIDE; ++r) {
        for (int c = 0; c < IMG_SIDE; ++c)
            std::cout << (s.pixels[r * IMG_SIDE + c] > 0.5 ? '#' : '.');
        std::cout << "\n";
    }
}
