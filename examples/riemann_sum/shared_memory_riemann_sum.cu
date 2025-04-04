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

template <typename T, T (*func)(T)>
__global__ void sharedMemoryRiemannSum(T a, T dx, int N, T *result) {
    __shared__ T sharedMem[BLOCK_SIZE];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    T sum = 0.0;

    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        T x = a + i * dx;
        sum += func(x) * dx;
    }

    sharedMem[tid] = sum;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s /= 2) {
        if (tid < s) {
            sharedMem[tid] += sharedMem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, sharedMem[0]);
    }
}


int main() {
    const double a = 0.1, b = 10.0;
    const int N = 1e7;

    double* result;
    cudaMallocManaged(&result, sizeof(double));

    /*
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


    */
    printf("\nGPU Results:\n");

    const int THREADS = 256;
    const int BLOCKS = (N + (THREADS - 1)) / THREADS;
    double dx = (b - a) / N;

    *result = 0.0;
    auto t1 = std::chrono::high_resolution_clock::now();
    sharedMemoryRiemannSum<double, sin_exp_gpu><<<BLOCKS, THREADS>>>(a, dx, N, result);
    cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();
    printf("sin_exp: %.10f (%.6f sec)\n", *result,
           std::chrono::duration<double>(t2 - t1).count());

    *result = 0.0;
    t1 = std::chrono::high_resolution_clock::now();
    sharedMemoryRiemannSum<double, square_gpu><<<BLOCKS, THREADS>>>(a, dx, N, result);
    cudaDeviceSynchronize();
    t2 = std::chrono::high_resolution_clock::now();
    printf("square: %.10f (%.6f sec)\n", *result,
           std::chrono::duration<double>(t2 - t1).count());

    *result = 0.0;
    t1 = std::chrono::high_resolution_clock::now();
    sharedMemoryRiemannSum<double, logarithm_exp_gpu><<<BLOCKS, THREADS>>>(a, dx, N, result);
    cudaDeviceSynchronize();
    t2 = std::chrono::high_resolution_clock::now();
    printf("log_exp: %.10f (%.6f sec)\n", *result,
           std::chrono::duration<double>(t2 - t1).count());

    *result = 0.0;
    t1 = std::chrono::high_resolution_clock::now();
    sharedMemoryRiemannSum<double, logarithm_sin_exp_gpu><<<BLOCKS, THREADS>>>(a, dx, N, result);
    cudaDeviceSynchronize();
    t2 = std::chrono::high_resolution_clock::now();
    printf("log_sin_exp: %.10f (%.6f sec)\n", *result,
           std::chrono::duration<double>(t2 - t1).count());

    *result = 0.0;
    t1 = std::chrono::high_resolution_clock::now();
    sharedMemoryRiemannSum<double, complex_kernel_gpu><<<BLOCKS, THREADS>>>(a, dx, N, result);
    cudaDeviceSynchronize();
    t2 = std::chrono::high_resolution_clock::now();
    printf("complex kernel: %.10f (%.6f sec)\n", *result,
           std::chrono::duration<double>(t2 - t1).count());

    cudaFree(result);
    return 0;
}

