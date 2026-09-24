CXX      ?= g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -pedantic -MMD -MP

SRC := $(wildcard src/*.cpp)
OBJ := $(SRC:src/%.cpp=build/%.o)
BIN := build/nn

.PHONY: all run compare clean

all: $(BIN)

$(BIN): $(OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@

build/%.o: src/%.cpp | build
	$(CXX) $(CXXFLAGS) -c $< -o $@

build:
	mkdir -p build

run: $(BIN)
	./$(BIN) demo

compare: $(BIN)
	./$(BIN) compare

clean:
	rm -rf build

-include $(OBJ:.o=.d)
