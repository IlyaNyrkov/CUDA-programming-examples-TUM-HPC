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

void print_array(int N, int* x) {
    printf("printing first %d elements: ", N);
    for (int i = 0; i < N; i++) {
        printf("%d ", x[i]);
    }
    printf("\n");
}

__global__ void blockSumAtomicAdd(int *in, int *globalSum, int n) {
    extern __shared__ int subArray[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    subArray[tid] = 0;

    if (gid < n) subArray[tid] += in[gid];
    if (gid + blockDim.x < n) subArray[tid] += in[gid + blockDim.x];
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            subArray[tid] += subArray[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(globalSum, subArray[0]);
    }
}

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : (1 << 28);  // Default: ~268M elements
    cout << "Summing " << N << " integers using GPU (Block-wise reduction with atomicAdd)\n";

    int *input, *globalSum;
    cudaMallocManaged(&input, N * sizeof(int));
    cudaMallocManaged(&globalSum, sizeof(int));
    *globalSum = 0;

    fill_array(N, input);

    const int THREAD_COUNT = BLOCK_SIZE;
    const int BLOCK_COUNT = (N + THREAD_COUNT * 2 - 1) / (THREAD_COUNT * 2);

    auto start_gpu = chrono::high_resolution_clock::now();
    blockSumAtomicAdd<<<BLOCK_COUNT, THREAD_COUNT, THREAD_COUNT * sizeof(int)>>>(input, globalSum, N);
    cudaDeviceSynchronize();
    auto end_gpu = chrono::high_resolution_clock::now();
    chrono::duration<double> gpu_time = end_gpu - start_gpu;

    cout << "GPU Array sum result: " << *globalSum << endl;
    cout << "Total GPU reduction time (atomicAdd only): " << gpu_time.count() << " seconds\n";

    // Cleanup
    cudaFree(input);
    cudaFree(globalSum);

    return 0;
}
