#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <chrono>
#define BLOCK_SIZE 256

// run with compilation flag -arch=sm_61 (atomicAdd for doubles)
// nvcc riemann_sum_bench_templ.cu -o riemann_sum -arch=sm_61

// Function pointer template
template<typename T>
using func_templ = T (*) (T);

// Define example functions
template <typename T>
__device__ T sin_exp(T x) {
    return exp(x) * sin(x);
}

template <typename T>
__device__ T square(T x) {
    return x * x;
}

template <typename T>
__device__ T logarithm_exp(T x) {
    return exp(__log2f(x));
}


template <typename T>
__device__ T logarithm_sin_exp(T x) {
   return __log2f(1 + x * x) * sin(exp(x));
}

// 1. Naive Implementation (Atomic)
template <typename T, T (*func)(T)>
__global__ void naiveRiemannSum(T a, T dx, int N, T *result) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    T sum = 0.0;

    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        T x = a + i * dx;
        sum += func(x) * dx;
    }

    atomicAdd(result, sum);
}

// 2. Grid-stride Loop Optimization
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

// 3. Loop Unrolling (2-way unrolling)
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

// 4. Warp Unrolling (Using Warp Shuffle)
template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
    for (int offset = 32 / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename T, T (*func)(T)>
__global__ void warpOptimizedRiemannSum(T a, T dx, int N, T *result) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    T sum = 0.0;

    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        T x = a + i * dx;
        sum += func(x) * dx;
    }

    sum = warpReduceSum(sum);

    if (threadIdx.x % 32 == 0) {
        atomicAdd(result, sum);
    }
}

// 5. Shared Memory Optimization (Block-Wise Reduction)
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

// **Host function to execute and measure time**
template <typename T, T (*func)(T), void (*kernel)(T, T, int, T *)>
void runKernel(const char *method, T a, T b, int N) {
    const int blocks = 256;
    const int threads = BLOCK_SIZE;
    T dx = (b - a) / N;

    T *d_result;
    cudaMalloc(&d_result, sizeof(T));
    cudaMemset(d_result, 0, sizeof(T));

    auto start = std::chrono::high_resolution_clock::now();
    kernel<<<blocks, threads>>>(a, dx, N, d_result);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<T> elapsed = end - start;

    T result;
    cudaMemcpy(&result, d_result, sizeof(T), cudaMemcpyDeviceToHost);
    printf("%s: Integral = %.10f, Time = %f sec\n", method, result, elapsed.count());

    cudaFree(d_result);
}

int main() {
    double a = 0, b = 10;  // Large interval
    int N = 1e9;  // Large iterations

    printf("Comparing Riemann Sum Implementations for Different Functions:\n");

    printf("\nIntegrating exp(x) * sin(x):\n");
    runKernel<double, sin_exp<double>, naiveRiemannSum<double, sin_exp<double>>>("Naïve", a, b, N);
    runKernel<double, sin_exp<double>, gridStrideRiemannSum<double, sin_exp<double>>>("Grid-Stride", a, b, N);
    runKernel<double, sin_exp<double>, unrolledRiemannSum<double, sin_exp<double>>>("Loop Unrolled", a, b, N);
    runKernel<double, sin_exp<double>, warpOptimizedRiemannSum<double, sin_exp<double>>>("Warp Optimized", a, b, N);
    runKernel<double, sin_exp<double>, sharedMemoryRiemannSum<double, sin_exp<double>>>("Shared Memory", a, b, N);

    printf("\nIntegrating x^2:\n");
    runKernel<double, square<double>, naiveRiemannSum<double, square<double>>>("Naïve", a, b, N);
    runKernel<double, square<double>, gridStrideRiemannSum<double, square<double>>>("Grid-Stride", a, b, N);
    runKernel<double, square<double>, unrolledRiemannSum<double, square<double>>>("Loop Unrolled", a, b, N);
    runKernel<double, square<double>, warpOptimizedRiemannSum<double, square<double>>>("Warp Optimized", a, b, N);
    runKernel<double, square<double>, sharedMemoryRiemannSum<double, square<double>>>("Shared Memory", a, b, N);

    printf("\nIntegrating exp(log2(x)):\n");
    runKernel<double, logarithm_exp<double>, naiveRiemannSum<double, logarithm_exp<double>>>("Naïve", a, b, N);
    runKernel<double, logarithm_exp<double>, gridStrideRiemannSum<double, logarithm_exp<double>>>("Grid-Stride", a, b, N);
    runKernel<double, logarithm_exp<double>, unrolledRiemannSum<double, logarithm_exp<double>>>("Loop Unrolled", a, b, N);
    runKernel<double, logarithm_exp<double>, warpOptimizedRiemannSum<double, logarithm_exp<double>>>("Warp Optimized", a, b, N);
    runKernel<double, logarithm_exp<double>, sharedMemoryRiemannSum<double, logarithm_exp<double>>>("Shared Memory", a, b, N);

    printf("\nIntegrating log2(1 + x * x) * sin(exp(x)):\n");
    runKernel<double, logarithm_sin_exp<double>, naiveRiemannSum<double, logarithm_sin_exp<double>>>("Naïve", a, b, N);
    runKernel<double, logarithm_sin_exp<double>, gridStrideRiemannSum<double, logarithm_sin_exp<double>>>("Grid-Stride", a, b, N);
    runKernel<double, logarithm_sin_exp<double>, unrolledRiemannSum<double, logarithm_sin_exp<double>>>("Loop Unrolled", a, b, N);
    runKernel<double, logarithm_sin_exp<double>, warpOptimizedRiemannSum<double, logarithm_sin_exp<double>>>("Warp Optimized", a, b, N);
    runKernel<double, logarithm_sin_exp<double>, sharedMemoryRiemannSum<double, logarithm_sin_exp<double>>>("Shared Memory", a, b, N);

    return 0;
}