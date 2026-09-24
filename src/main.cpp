#include <iostream>
#include <string>

int main(int argc, char **argv) {
    std::string mode = argc > 1 ? argv[1] : "demo";

    if (mode == "demo") {
        std::cout << "mode: demo (stub)\n";
    } else if (mode == "compare") {
        std::cout << "mode: compare (stub)\n";
    } else {
        std::cerr << "usage: nn [demo|compare]\n";
        return 1;
    }
    return 0;
}
