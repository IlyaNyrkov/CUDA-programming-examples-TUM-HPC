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


int main(int argc, char* argv[]) {
    const double a = 0.1, b = 10.0;
    int N = (argc > 1) ? atoi(argv[1]) : 100000;  // Default to 10^5

    double* d_result;
    cudaMalloc(&d_result, sizeof(double));
    cudaMemset(d_result, 0, sizeof(double));

    const int THREADS = BLOCK_SIZE;
    const int BLOCKS = (N + THREADS - 1) / THREADS;
    double dx = (b - a) / N;

    auto start = std::chrono::high_resolution_clock::now();
    sharedMemoryRiemannSum<double, complex_kernel_gpu><<<BLOCKS, THREADS>>>(a, dx, N, d_result);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    double host_result = 0;
    cudaMemcpy(&host_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    printf("complex kernel result: %.10f\n", host_result);
    printf("execution time        : %.6f seconds\n",
           std::chrono::duration<double>(end - start).count());

    cudaFree(d_result);
    return 0;
}