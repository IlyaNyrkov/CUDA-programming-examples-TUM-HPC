#include <stdio.h>
#include <iostream>
#include <chrono>

using namespace std;

#define BLOCK_SIZE 256   // Optimal for L40S (also try 512)

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
__global__ void gridStrideReductionAtomic(int *in, int *globalSum, int n) {
    extern __shared__ int subArr[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    int gridSize = blockDim.x * gridDim.x * 2;

    int localSum = 0;
    while (gid < n) {
        localSum += in[gid];
        if (gid + blockSize < n)
            localSum += in[gid + blockSize];
        gid += gridSize;
    }

    subArr[tid] = localSum;
    __syncthreads();

    if (blockSize >= 512) { if (tid < 256) subArr[tid] += subArr[tid + 256]; __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) subArr[tid] += subArr[tid + 128]; __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64)  subArr[tid] += subArr[tid + 64];  __syncthreads(); }

    if (tid < 32) warpReduceTemplate<blockSize>(subArr, tid);

    if (tid == 0) atomicAdd(globalSum, subArr[0]);
}

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : (1 << 28);  // Default: 2^28
    cout << "Summing " << N << " integers using GPU (Grid-stride + Warp unroll + atomicAdd)\n";

    // Allocate and fill
    int *input, *globalSum;
    cudaMallocManaged(&input, N * sizeof(int));
    cudaMallocManaged(&globalSum, sizeof(int));
    *globalSum = 0;
    fill_array(N, input);

    const int THREAD_COUNT = BLOCK_SIZE;
    const int BLOCK_COUNT = (N + THREAD_COUNT * 2 - 1) / (THREAD_COUNT * 2);

    auto start_gpu = chrono::high_resolution_clock::now();
    gridStrideReductionAtomic<THREAD_COUNT><<<BLOCK_COUNT, THREAD_COUNT, THREAD_COUNT * sizeof(int)>>>(input, globalSum, N);
    cudaDeviceSynchronize();
    auto end_gpu = chrono::high_resolution_clock::now();

    chrono::duration<double> gpu_time = end_gpu - start_gpu;

    cout << "GPU Array sum result        : " << *globalSum << endl;
    cout << "GPU reduction total time    : " << gpu_time.count() << " seconds\n";

    cudaFree(input);
    cudaFree(globalSum);

    return 0;
}
