#include "pca.hpp"


void print_times(std::array<int64_t, 5> times)
{
    std::cout << "| Times |" << std::endl;
    std::cout << "Fit: " << times[0] << "ms\n";    
    std::cout << "Transform: " << times[1] << "ms\n";
    std::cout << "Covariance Matrix: " << times[2] << "s\n";
    std::cout << "Eigen Decomposition: " << times[3] << "s\n";
    std::cout << "Projection: " << times[4] << "ms\n";
}