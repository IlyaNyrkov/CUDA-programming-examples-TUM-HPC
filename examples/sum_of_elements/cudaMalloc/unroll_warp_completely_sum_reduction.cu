#include <stdio.h>
#include <iostream>
#include <chrono>
#include <cstdlib>

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
__global__ void unrollWarpCompletelyReduction(int *input, int *partialSums, int n) {
    extern __shared__ int subArray[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    int sum = 0;
    if (gid < n) sum += input[gid];
    if (gid + blockDim.x < n) sum += input[gid + blockDim.x];

    subArray[tid] = sum;
    __syncthreads();

    if (blockSize >= 128) { if (tid < 64) subArray[tid] += subArray[tid + 64]; __syncthreads(); }

    if (tid < 32) warpReduceTemplate<blockSize>(subArray, tid);

    if (tid == 0) {
        partialSums[blockIdx.x] = subArray[0];
    }
}

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : (1 << 28);  // Default: 268M elements

    if (N <= 0) {
        cerr << "Invalid number of elements.\n";
        return 1;
    }

    cout << "Summing " << N << " integers using Warp-unrolled reduction (GPU-only)\n";

    // Host allocation and fill
    int* h_input = (int*)malloc(N * sizeof(int));
    fill_array(N, h_input);

    // Device allocation
    int *d_input, *d_partialSums;
    cudaMalloc(&d_input, N * sizeof(int));

    const int THREAD_COUNT = BLOCK_SIZE;
    const int BLOCK_COUNT = (N + THREAD_COUNT * 2 - 1) / (THREAD_COUNT * 2);
    cudaMalloc(&d_partialSums, BLOCK_COUNT * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel and measure time
    auto start_gpu = chrono::high_resolution_clock::now();
    unrollWarpCompletelyReduction<THREAD_COUNT><<<BLOCK_COUNT, THREAD_COUNT, THREAD_COUNT * sizeof(int)>>>(d_input, d_partialSums, N);
    cudaDeviceSynchronize();

    // Copy results back
    int* h_partialSums = (int*)malloc(BLOCK_COUNT * sizeof(int));
    cudaMemcpy(h_partialSums, d_partialSums, BLOCK_COUNT * sizeof(int), cudaMemcpyDeviceToHost);

    // Final reduction on CPU
    long long final_sum = 0;
    for (int i = 0; i < BLOCK_COUNT; i++) {
        final_sum += h_partialSums[i];
    }
    auto end_gpu = chrono::high_resolution_clock::now();
    chrono::duration<double> gpu_time = end_gpu - start_gpu;

    // Print result
    cout << "GPU Array sum result      : " << final_sum << endl;
    cout << "GPU total reduction time  : " << gpu_time.count() << " seconds\n";

    // Cleanup
    free(h_input);
    free(h_partialSums);
    cudaFree(d_input);
    cudaFree(d_partialSums);

    return 0;
}
