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

__global__ void blockSumReduction(int *in, int *out, int n) {
    extern __shared__ int subArray[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    subArray[tid] = 0;
    if (gid < n) subArray[tid] += in[gid];
    if (gid + blockDim.x < n) subArray[tid] += in[gid + blockDim.x];
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            subArray[tid] += subArray[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = subArray[0]; // Write per-block sum
    }
}

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : (1 << 28);  // Default: 268M ints

    cout << "Summing " << N << " integers using GPU (Block-wise shared memory + final CPU sum)\n";

    // --- Host allocation & initialization
    int *h_input = (int*)malloc(N * sizeof(int));
    fill_array(N, h_input);

    // --- Device allocations
    int *d_input, *d_partial_sums;
    cudaMalloc(&d_input, N * sizeof(int));

    const int THREAD_COUNT = BLOCK_SIZE;
    const int BLOCK_COUNT = (N + THREAD_COUNT * 2 - 1) / (THREAD_COUNT * 2);

    cudaMalloc(&d_partial_sums, BLOCK_COUNT * sizeof(int));

    // --- Copy data to device
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    // --- GPU Kernel Launch & Timing
    auto start_gpu = chrono::high_resolution_clock::now();
    blockSumReduction<<<BLOCK_COUNT, THREAD_COUNT, THREAD_COUNT * sizeof(int)>>>(d_input, d_partial_sums, N);
    cudaDeviceSynchronize();
    auto end_gpu = chrono::high_resolution_clock::now();

    // --- Copy back partial sums
    int *h_partial_sums = (int*)malloc(BLOCK_COUNT * sizeof(int));
    cudaMemcpy(h_partial_sums, d_partial_sums, BLOCK_COUNT * sizeof(int), cudaMemcpyDeviceToHost);

    long long final_sum = 0;
    for (int i = 0; i < BLOCK_COUNT; i++) {
        final_sum += h_partial_sums[i];
    }

    chrono::duration<double> gpu_time = end_gpu - start_gpu;

    // --- Output
    cout << "GPU Array sum result       : " << final_sum << endl;
    cout << "Total GPU reduction time   : " << gpu_time.count() << " seconds\n";

    // --- Cleanup
    free(h_input);
    free(h_partial_sums);
    cudaFree(d_input);
    cudaFree(d_partial_sums);

    return 0;
}
