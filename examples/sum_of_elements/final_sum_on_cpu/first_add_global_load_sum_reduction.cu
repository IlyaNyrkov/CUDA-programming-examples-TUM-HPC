#include <stdio.h>
#include <iostream>
#include <chrono>

using namespace std;

#define BLOCK_SIZE 256  // L40S/H100 optimal: 256–512

void fill_array(int N, int* x) {
    for (int i = 0; i < N; i++) {
        x[i] = 1;
    }
}

__global__ void firstAddGlobalLoadReduction(int *in, int *partialSums, int n) {
    extern __shared__ int subArr[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    int val1 = (gid < n) ? in[gid] : 0;
    int val2 = (gid + blockDim.x < n) ? in[gid + blockDim.x] : 0;

    subArr[tid] = val1 + val2;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            subArr[tid] += subArr[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partialSums[blockIdx.x] = subArr[0];
    }
}

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : (1 << 28);  // Default: 268M elements

    cout << "Summing " << N << " integers using First-Add-Global-Load (GPU-only + CPU final sum)\n";

    // Allocate and fill data
    int *input;
    cudaMallocManaged(&input, N * sizeof(int));
    fill_array(N, input);

    // Kernel config
    const int THREAD_COUNT = BLOCK_SIZE;
    const int BLOCK_COUNT = (N + THREAD_COUNT * 2 - 1) / (THREAD_COUNT * 2);

    int *partial_sums;
    cudaMallocManaged(&partial_sums, BLOCK_COUNT * sizeof(int));

    // GPU benchmark (includes CPU final sum)
    auto start_gpu = chrono::high_resolution_clock::now();
    firstAddGlobalLoadReduction<<<BLOCK_COUNT, THREAD_COUNT, THREAD_COUNT * sizeof(int)>>>(input, partial_sums, N);
    cudaDeviceSynchronize();

    long long final_sum = 0;
    for (int i = 0; i < BLOCK_COUNT; i++) {
        final_sum += partial_sums[i];
    }
    auto end_gpu = chrono::high_resolution_clock::now();
    chrono::duration<double> gpu_time = end_gpu - start_gpu;

    cout << "GPU Array sum result       : " << final_sum << endl;
    cout << "GPU total reduction time   : " << gpu_time.count() << " seconds\n";

    // Cleanup
    cudaFree(input);
    cudaFree(partial_sums);

    return 0;
}
