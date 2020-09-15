#include <engine.h>
#include <iostream>

#include <MatlabInterface.h>
#include <MatlabInterface.cpp>

constexpr const int rows = 2;
constexpr const int cols = 2;
constexpr const int size = rows * cols;

int main() {
    // std::cout << engOpen("\0") << std::endl;
    double mat[size] = { 1, 2, 3, 4 };
    MatlabInterface::GetEngine().SetEngineRealMatrix("asdf", rows, cols, mat);

    return 0;
}