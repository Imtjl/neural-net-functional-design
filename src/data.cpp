#include "data.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

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

void printGrid(const std::vector<const Sample *> &samples,
               const std::vector<std::string> &captions) {
    const int PER_ROW = 4, WIDTH = 23;
    for (size_t first = 0; first < samples.size(); first += PER_ROW) {
        size_t last = std::min(samples.size(), first + PER_ROW);
        auto cell = [&](const std::string &text) { // left-aligned column
            std::string t = text.substr(0, WIDTH - 1);
            std::cout << t << std::string(WIDTH - t.size(), ' ');
        };
        for (size_t i = first; i < last; ++i)
            cell(samples[i]->name);
        std::cout << "\n";
        for (size_t i = first; i < last; ++i)
            cell(captions[i]);
        std::cout << "\n";
        for (int r = 0; r < IMG_SIDE; ++r) {
            for (size_t i = first; i < last; ++i) {
                std::string row;
                for (int c = 0; c < IMG_SIDE; ++c)
                    row += samples[i]->pixels[r * IMG_SIDE + c] > 0.5 ? '#' : '.';
                cell(row);
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}
