#include <stdio.h> 
#include <math.h>
#include <chrono>

#define BLOCK_SIZE 256

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

template <unsigned int blockSize, typename T>
__device__ void warpReduceUnrolled(volatile T* sdata, int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <typename T, T (*func)(T), unsigned int blockSize>
__global__ void optimizedRiemannSum(T a, T dx, int N, T *partial_sums) {
    extern __shared__ T subArray[];

    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x * 2;
    int stride = blockDim.x * gridDim.x * 2;

    T local_sum = 0.0;

    for (int i = idx; i < N; i += stride) {
        T x1 = a + i * dx;
        T x2 = a + (i + blockDim.x) * dx;
        local_sum += func(x1) * dx;
        if (i + blockDim.x < N) {
            local_sum += func(x2) * dx;
        }
    }

    subArray[tid] = local_sum;
    __syncthreads();

    if (blockSize >= 512) { if (tid < 256) subArray[tid] += subArray[tid + 256]; __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) subArray[tid] += subArray[tid + 128]; __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64)  subArray[tid] += subArray[tid + 64];  __syncthreads(); }

    if (tid < 32) warpReduceUnrolled<blockSize, T>(subArray, tid);

    if (tid == 0) partial_sums[blockIdx.x] = subArray[0];
}


int main(int argc, char* argv[]) {
    const double a = 0.1, b = 10.0;
    int N = (argc > 1) ? atoi(argv[1]) : 100000000;

    const int THREADS = BLOCK_SIZE;
    const int BLOCKS = (N + THREADS * 2 - 1) / (THREADS * 2);
    double dx = (b - a) / N;

    double *d_partial_sums, *h_partial_sums;
    cudaMalloc(&d_partial_sums, BLOCKS * sizeof(double));
    h_partial_sums = (double*)malloc(BLOCKS * sizeof(double));

    auto start = std::chrono::high_resolution_clock::now();
    optimizedRiemannSum<double, complex_kernel_gpu, BLOCK_SIZE>
        <<<BLOCKS, THREADS, THREADS * sizeof(double)>>>(a, dx, N, d_partial_sums);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_partial_sums, d_partial_sums, BLOCKS * sizeof(double), cudaMemcpyDeviceToHost);

    double final_result = 0.0;
    for (int i = 0; i < BLOCKS; ++i) {
        final_result += h_partial_sums[i];
    }

    printf("complex kernel result: %.10f\n", final_result);
    printf("execution time        : %.6f seconds\n",
           std::chrono::duration<double>(end - start).count());

    cudaFree(d_partial_sums);
    free(h_partial_sums);
    return 0;
}