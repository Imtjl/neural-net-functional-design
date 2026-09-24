CXX      ?= g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -pedantic -MMD -MP

SRC := $(wildcard src/*.cpp)
OBJ := $(SRC:src/%.cpp=build/%.o)
BIN := build/nn

.PHONY: all run compare alpha figures clean

all: $(BIN)

$(BIN): $(OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@

build/%.o: src/%.cpp | build
	$(CXX) $(CXXFLAGS) -c $< -o $@

build:
	mkdir -p build

run: $(BIN)
	./$(BIN) demo $(S)

compare: $(BIN)
	./$(BIN) compare $(E)

alpha: $(BIN)
	./$(BIN) alpha $(S)

figures: $(BIN)
	python3 tools/report_figures.py

clean:
	rm -rf build

-include $(OBJ:.o=.d)
