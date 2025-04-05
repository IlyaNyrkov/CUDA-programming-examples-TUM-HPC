#include <stdio.h>
#include <iostream>
#include <chrono>

using namespace std;

#define BLOCK_SIZE 256  // L40S/H100 optimal: 256–512

void fill_array(int N, int* x) {
    for (int i = 0; i < N; i++) {
        x[i] = rand() % 10;
    }
}

template <unsigned int blockSize>
__device__ void warpReduceTemplate(volatile int* s, int tid) {
    if (blockSize >= 64) s[tid] += s[tid + 32];
    if (blockSize >= 32) s[tid] += s[tid + 16];
    if (blockSize >= 16) s[tid] += s[tid + 8];
    if (blockSize >= 8)  s[tid] += s[tid + 4];
    if (blockSize >= 4)  s[tid] += s[tid + 2];
    if (blockSize >= 2)  s[tid] += s[tid + 1];
}

template <unsigned int blockSize>
__global__ void unrollWarpCompletelyReduction_atomic(int *input, int *output, int n) {
    extern __shared__ int subArray[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    int sum = 0;
    if (gid < n) sum += input[gid];
    if (gid + blockDim.x < n) sum += input[gid + blockDim.x];

    subArray[tid] = sum;
    __syncthreads();

    if (blockSize >= 512) { if (tid < 256) subArray[tid] += subArray[tid + 256]; __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) subArray[tid] += subArray[tid + 128]; __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64)  subArray[tid] += subArray[tid + 64];  __syncthreads(); }

    if (tid < 32) warpReduceTemplate<blockSize>(subArray, tid);

    if (tid == 0) atomicAdd(output, subArray[0]);
}

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : (1 << 28);  // Default: 2^28 ~268M

    if (N <= 0) {
        cerr << "Invalid number of elements.\n";
        return 1;
    }

    cout << "Summing " << N << " integers using warp-unrolled reduction with atomicAdd...\n";

    // Allocate unified memory
    int *input, *output_gpu;
    cudaMallocManaged(&input, N * sizeof(int));
    cudaMallocManaged(&output_gpu, sizeof(int));
    *output_gpu = 0;

    fill_array(N, input);

    const int THREAD_COUNT = BLOCK_SIZE;
    const int BLOCK_COUNT = (N + THREAD_COUNT * 2 - 1) / (THREAD_COUNT * 2);

    // Run GPU kernel
    *output_gpu = 0;
    auto start_gpu = chrono::high_resolution_clock::now();
    unrollWarpCompletelyReduction_atomic<THREAD_COUNT><<<BLOCK_COUNT, THREAD_COUNT, THREAD_COUNT * sizeof(int)>>>(input, output_gpu, N);
    cudaDeviceSynchronize();
    auto end_gpu = chrono::high_resolution_clock::now();
    chrono::duration<double> gpu_time = end_gpu - start_gpu;

    cout << "GPU Array sum result: " << *output_gpu << endl;
    cout << "GPU Array sum time  : " << gpu_time.count() << " seconds\n";

    cudaFree(input);
    cudaFree(output_gpu);

    return 0;
}
