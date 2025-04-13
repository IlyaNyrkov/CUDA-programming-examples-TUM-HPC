#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <iostream>
#include <cmath>
#include <chrono>

// Complex function as a functor
struct ComplexRiemannFunction {
    double a, dx;

    __host__ __device__
    ComplexRiemannFunction(double a_, double dx_) : a(a_), dx(dx_) {}

    __host__ __device__
    double operator()(int i) const {
        double x = a + i * dx;
        double inner = sin(exp(x) + log(x + 1.0)) + sqrt(x * x + 1.0);
        double fx = pow(inner, 2.5) * cos(5.0 * x) / (1.0 + exp(-x));
        return fx * dx;
    }
};

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : 1000000000;  // Default: 1e9
    double a = 0.1, b = 10.0;
    double dx = (b - a) / N;

    auto begin = thrust::counting_iterator<int>(0);
    auto end = begin + N;

    ComplexRiemannFunction func(a, dx);

    auto start = std::chrono::high_resolution_clock::now();

    double result = thrust::transform_reduce(
        begin, end,
        func,
        0.0,
        thrust::plus<double>()
    );

    auto end_time = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end_time - start).count();

    std::cout << "Riemann sum of complex function: " << result << "\n";
    std::cout << "Execution time: " << duration << " seconds\n";

    return 0;
}