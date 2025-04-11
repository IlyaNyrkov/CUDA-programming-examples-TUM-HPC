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

template <typename T>
__device__ void warpReduceUnrolled(volatile T* s, int tid) {
    s[tid] += s[tid + 32];
    s[tid] += s[tid + 16];
    s[tid] += s[tid + 8];
    s[tid] += s[tid + 4];
    s[tid] += s[tid + 2];
    s[tid] += s[tid + 1];
}

template <typename T, T (*func)(T)>
__global__ void warpOptimizedRiemannSum(T a, T dx, int N, T *partial_sums) {
    extern __shared__ T shared[];

    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    T local_sum = 0.0;
    for (int i = idx; i < N; i += stride) {
        T x = a + i * dx;
        local_sum += func(x) * dx;
    }

    shared[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }

    if (tid < 32) {
        warpReduceUnrolled(shared, tid);
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = shared[0];
    }
}


int main(int argc, char* argv[]) {
    const double a = 0.1, b = 10.0;
    int N = (argc > 1) ? atoi(argv[1]) : 10000000;

    const int THREADS = BLOCK_SIZE;
    const int BLOCKS = (N + THREADS - 1) /  THREADS;
    double dx = (b - a) / N;

    // Allocate space for partial sums from each block
    double *d_partial_sums, *h_partial_sums;
    cudaMalloc(&d_partial_sums, BLOCKS * sizeof(double));
    h_partial_sums = (double*)malloc(BLOCKS * sizeof(double));

    auto start = std::chrono::high_resolution_clock::now();
    warpOptimizedRiemannSum<double, complex_kernel_gpu>
        <<<BLOCKS, THREADS, THREADS * sizeof(double)>>>(a, dx, N, d_partial_sums);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    // Copy partial results and do final sum on CPU
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