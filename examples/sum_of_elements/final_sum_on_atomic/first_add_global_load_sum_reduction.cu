#include <stdio.h>
#include <iostream>
#include <chrono>

using namespace std;

#define BLOCK_SIZE 256  // Good choice for L40S

void fill_array(int N, int* x) {
    for (int i = 0; i < N; i++) {
        x[i] = rand() % 10;
    }
}

__global__ void firstAddGlobalLoadAtomic(int *in, int *globalSum, int n) {
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
        atomicAdd(globalSum, subArr[0]);
    }
}

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : (1 << 28);  // Default: 2^28 = ~268M elements
    cout << "Summing " << N << " integers using GPU (First Add During Global Load with atomicAdd)\n";

    // Allocate unified memory
    int *input, *globalSum;
    cudaMallocManaged(&input, N * sizeof(int));
    cudaMallocManaged(&globalSum, sizeof(int));
    *globalSum = 0;

    fill_array(N, input);

    // Launch configuration
    const int THREAD_COUNT = BLOCK_SIZE;
    const int BLOCK_COUNT = (N + THREAD_COUNT * 2 - 1) / (THREAD_COUNT * 2);

    // --- GPU SUM ---
    auto start_gpu = chrono::high_resolution_clock::now();
    firstAddGlobalLoadAtomic<<<BLOCK_COUNT, THREAD_COUNT, THREAD_COUNT * sizeof(int)>>>(input, globalSum, N);
    cudaDeviceSynchronize();
    auto end_gpu = chrono::high_resolution_clock::now();
    chrono::duration<double> gpu_time = end_gpu - start_gpu;

    cout << "GPU Array sum result        : " << *globalSum << endl;
    cout << "GPU reduction total time    : " << gpu_time.count() << " seconds\n";

    // Cleanup
    cudaFree(input);
    cudaFree(globalSum);

    return 0;
}
