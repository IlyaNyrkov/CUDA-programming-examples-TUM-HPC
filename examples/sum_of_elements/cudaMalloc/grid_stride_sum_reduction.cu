#include <stdio.h>
#include <iostream>
#include <chrono>
#include <cstdlib>

using namespace std;

#define BLOCK_SIZE 256  // Optimal for H100: 256–512

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
__global__ void gridStrideReduction(int *input, int *partialSums, int n) {
    extern __shared__ int shared[];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    int gridSize = blockDim.x * gridDim.x * 2;

    int sum = 0;
    while (index < n) {
        sum += input[index];
        if (index + blockSize < n)
            sum += input[index + blockSize];
        index += gridSize;
    }

    shared[tid] = sum;
    __syncthreads();

    if (blockSize >= 512) { if (tid < 256) shared[tid] += shared[tid + 256]; __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) shared[tid] += shared[tid + 128]; __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64)  shared[tid] += shared[tid + 64];  __syncthreads(); }

    if (tid < 32) warpReduceTemplate<blockSize>(shared, tid);

    if (tid == 0) partialSums[blockIdx.x] = shared[0];
}

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : (1 << 28);  // Default: 268M elements
    cout << "Summing " << N << " integers using Grid-stride + Warp unroll (GPU only)\n";

    // --- Host memory allocation
    int* h_input = (int*)malloc(N * sizeof(int));
    fill_array(N, h_input);

    // --- Device memory allocation
    int *d_input, *d_partialSums;
    cudaMalloc(&d_input, N * sizeof(int));

    const int THREAD_COUNT = BLOCK_SIZE;
    const int BLOCK_COUNT = (N + THREAD_COUNT * 2 - 1) / (THREAD_COUNT * 2);
    cudaMalloc(&d_partialSums, BLOCK_COUNT * sizeof(int));

    // Copy input from host to device
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    // --- Run kernel
    auto start_gpu = chrono::high_resolution_clock::now();
    gridStrideReduction<THREAD_COUNT><<<BLOCK_COUNT, THREAD_COUNT, THREAD_COUNT * sizeof(int)>>>(d_input, d_partialSums, N);
    cudaDeviceSynchronize();

    // --- Copy back results
    int* h_partialSums = (int*)malloc(BLOCK_COUNT * sizeof(int));
    cudaMemcpy(h_partialSums, d_partialSums, BLOCK_COUNT * sizeof(int), cudaMemcpyDeviceToHost);

    // --- Final reduction on CPU
    long long gpu_result = 0;
    for (int i = 0; i < BLOCK_COUNT; ++i) {
        gpu_result += h_partialSums[i];
    }
    auto end_gpu = chrono::high_resolution_clock::now();
    chrono::duration<double> gpu_time = end_gpu - start_gpu;

    cout << "GPU Array sum result       : " << gpu_result << endl;
    cout << "GPU total reduction time   : " << gpu_time.count() << " seconds\n";

    // --- Cleanup
    free(h_input);
    free(h_partialSums);
    cudaFree(d_input);
    cudaFree(d_partialSums);

    return 0;
}
