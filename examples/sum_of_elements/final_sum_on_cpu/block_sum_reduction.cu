#include <stdio.h>
#include <iostream>
#include <chrono>

using namespace std;

#define BLOCK_SIZE 256  // Tune for your GPU (256–512 is good for L40S)

void fill_array(int N, int* x) {
    for (int i = 0; i < N; i++) {
        x[i] = 1;
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

    // Allocate and fill input
    int *input;
    cudaMallocManaged(&input, N * sizeof(int));
    fill_array(N, input);

    // --- GPU Kernel Launch ---
    const int THREAD_COUNT = BLOCK_SIZE;
    const int BLOCK_COUNT = (N + THREAD_COUNT * 2 - 1) / (THREAD_COUNT * 2);

    int *partial_sums;
    cudaMallocManaged(&partial_sums, BLOCK_COUNT * sizeof(int));

    auto start_gpu = chrono::high_resolution_clock::now();
    blockSumReduction<<<BLOCK_COUNT, THREAD_COUNT, THREAD_COUNT * sizeof(int)>>>(input, partial_sums, N);
    cudaDeviceSynchronize();

    // Final CPU sum of partial results
    long long final_sum = 0;
    for (int i = 0; i < BLOCK_COUNT; i++) {
        final_sum += partial_sums[i];
    }
    auto end_gpu = chrono::high_resolution_clock::now();
    chrono::duration<double> gpu_time = end_gpu - start_gpu;

    // Output
    cout << "GPU Array sum result       : " << final_sum << endl;
    cout << "Total GPU reduction time   : " << gpu_time.count() << " seconds\n";

    // Cleanup
    cudaFree(input);
    cudaFree(partial_sums);

    return 0;
}
