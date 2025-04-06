// -arch=sm_60
#include <stdio.h> 
#include <math.h>
#include <chrono>

#define BLOCK_SIZE 256

// CPU-side functions
double sin_exp(double x) {
    return exp(x) * sin(x);
}
double square(double x) {
    return x * x;
}
double logarithm_exp(double x) {
    return exp(log2(x));
}
double logarithm_sin_exp(double x) {
    return log2(1 + x * x) * sin(exp(x));
}

template <typename T>
void left_riemann_cpu(T (*func)(T), T a, T b, int iterations, T* result) {
    T dx = (b - a) / iterations;
    *result = 0.0;
    for (int i = 0; i < iterations; ++i) {
        T x = a + i * dx;
        *result += func(x) * dx;
    }
}

// GPU-side functions
template <typename T>
__device__ T sin_exp_gpu(T x) { return exp(x) * sin(x); }

template <typename T>
__device__ T square_gpu(T x) { return x * x; }

template <typename T>
__device__ T logarithm_exp_gpu(T x) { return exp(log2(x)); }

template <typename T>
__device__ T logarithm_sin_exp_gpu(T x) {
    return log2(1 + x * x) * sin(exp(x));
}

template <typename T>
__device__ T complex_kernel_gpu(T x) {
    return pow(sin(exp(x) + log(x + 1.0)) + sqrt(x * x + 1.0), 2.5) * cos(5.0 * x) / (1.0 + exp(-x));
}

template <typename T>
using func_templ = T (*)(T);

template <typename T, func_templ<T> func>
__global__ void naiveRiemannSum(T a, T dx, int iterations, T *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    T sum = 0.0;

    if (idx < iterations)  {
        T x = a + idx * dx;
        atomicAdd(result, func(x) * dx);
    }

}

int main(int argc, char* argv[]) {
    const double a = 0.1, b = 10.0;
    int N = (argc > 1) ? atoi(argv[1]) : 100000;  // Default: 10^5
    double dx = (b - a) / N;

    // Allocate device memory
    double* result_device;
    cudaMalloc(&result_device, sizeof(double));
    cudaMemset(result_device, 0, sizeof(double));

    const int THREADS = BLOCK_SIZE;
    const int BLOCKS = (N + THREADS - 1) / THREADS;

    auto t1 = std::chrono::high_resolution_clock::now();
    naiveRiemannSum<double, complex_kernel_gpu><<<BLOCKS, THREADS>>>(a, dx, N, result_device);
    cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();

    double result_host;
    cudaMemcpy(&result_host, result_device, sizeof(double), cudaMemcpyDeviceToHost);
    printf("complex kernel: %.10f (%.6f sec)\n", result_host,
           std::chrono::duration<double>(t2 - t1).count());

    cudaFree(result_device);
    return 0;
}