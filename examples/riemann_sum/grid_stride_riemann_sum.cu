// --compiler-args=-arch=sm_60
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

template <typename T, T (*func)(T)>
__global__ void gridStrideRiemannSum(T a, T dx, int N, T *result) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    T sum = 0.0;

    for (int i = idx; i < N; i += gridDim.x * blockDim.x) {
        T x = a + i * dx;
        sum += func(x) * dx;
    }

    atomicAdd(result, sum);
}

int main(int argc, char* argv[]) {
    const double a = 0.1, b = 10.0;
    int N = (argc > 1) ? atoi(argv[1]) : 10000000;

    double* d_result;
    cudaMalloc(&d_result, sizeof(double));
    cudaMemset(d_result, 0, sizeof(double));

    const int THREADS = BLOCK_SIZE;
    const int BLOCKS = (N + THREADS - 1) / THREADS;
    double dx = (b - a) / N;

    auto t1 = std::chrono::high_resolution_clock::now();
    gridStrideRiemannSum<double, complex_kernel_gpu><<<BLOCKS, THREADS>>>(a, dx, N, d_result);
    cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();

    double h_result;
    cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

    printf("complex kernel result: %.10f\n", h_result);
    printf("execution time        : %.6f seconds\n",
           std::chrono::duration<double>(t2 - t1).count());

    cudaFree(d_result);
    return 0;
}