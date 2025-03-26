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
using func_templ = T (*)(T);

template <typename T, T (*func)(T)>
__global__ void unrolledRiemannSum(T a, T dx, int N, T *result) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    T sum = 0.0;

    for (int i = idx; i < N; i += blockDim.x * gridDim.x * 2) {
        T x1 = a + i * dx;
        T x2 = a + (i + blockDim.x * gridDim.x) * dx;
        sum += func(x1) * dx;
        if (i + blockDim.x * gridDim.x < N) {
            sum += func(x2) * dx;
        }
    }

    atomicAdd(result, sum);
}


int main() {
    const double a = 0.0, b = 10.0;
    const int N = 1e7;

    double* result;
    cudaMallocManaged(&result, sizeof(double));

    printf("CPU Results:\n");

    double cpu_result;
    auto t1 = std::chrono::high_resolution_clock::now();
    left_riemann_cpu(sin_exp, a, b, N, &cpu_result);
    auto t2 = std::chrono::high_resolution_clock::now();
    printf("sin_exp: %.10f (%.6f sec)\n", cpu_result,
           std::chrono::duration<double>(t2 - t1).count());

    t1 = std::chrono::high_resolution_clock::now();
    left_riemann_cpu(square, a, b, N, &cpu_result);
    t2 = std::chrono::high_resolution_clock::now();
    printf("square: %.10f (%.6f sec)\n", cpu_result,
           std::chrono::duration<double>(t2 - t1).count());

    t1 = std::chrono::high_resolution_clock::now();
    left_riemann_cpu(logarithm_exp, a, b, N, &cpu_result);
    t2 = std::chrono::high_resolution_clock::now();
    printf("log_exp: %.10f (%.6f sec)\n", cpu_result,
           std::chrono::duration<double>(t2 - t1).count());

    t1 = std::chrono::high_resolution_clock::now();
    left_riemann_cpu(logarithm_sin_exp, a, b, N, &cpu_result);
    t2 = std::chrono::high_resolution_clock::now();
    printf("log_sin_exp: %.10f (%.6f sec)\n", cpu_result,
           std::chrono::duration<double>(t2 - t1).count());

    printf("\nGPU Results:\n");

    const int THREADS = 256;
    const int BLOCKS = (N + (THREADS - 1)) / THREADS;
    double dx = (b - a) / N;

    *result = 0.0;
    t1 = std::chrono::high_resolution_clock::now();
    unrolledRiemannSum<double, sin_exp_gpu><<<BLOCKS, THREADS>>>(a, dx, N, result);
    cudaDeviceSynchronize();
    t2 = std::chrono::high_resolution_clock::now();
    printf("sin_exp: %.10f (%.6f sec)\n", *result,
           std::chrono::duration<double>(t2 - t1).count());

    *result = 0.0;
    t1 = std::chrono::high_resolution_clock::now();
    unrolledRiemannSum<double, square_gpu><<<BLOCKS, THREADS>>>(a, dx, N, result);
    cudaDeviceSynchronize();
    t2 = std::chrono::high_resolution_clock::now();
    printf("square: %.10f (%.6f sec)\n", *result,
           std::chrono::duration<double>(t2 - t1).count());

    *result = 0.0;
    t1 = std::chrono::high_resolution_clock::now();
    unrolledRiemannSum<double, logarithm_exp_gpu><<<BLOCKS, THREADS>>>(a, dx, N, result);
    cudaDeviceSynchronize();
    t2 = std::chrono::high_resolution_clock::now();
    printf("log_exp: %.10f (%.6f sec)\n", *result,
           std::chrono::duration<double>(t2 - t1).count());

    *result = 0.0;
    t1 = std::chrono::high_resolution_clock::now();
    unrolledRiemannSum<double, logarithm_sin_exp_gpu><<<BLOCKS, THREADS>>>(a, dx, N, result);
    cudaDeviceSynchronize();
    t2 = std::chrono::high_resolution_clock::now();
    printf("log_sin_exp: %.10f (%.6f sec)\n", *result,
           std::chrono::duration<double>(t2 - t1).count());

    cudaFree(result);
    return 0;
}

