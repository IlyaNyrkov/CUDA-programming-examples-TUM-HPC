#include <stdio.h>
#include <iostream>
#include <chrono>

using namespace std;

#define BLOCK_SIZE 256   // Recommended for L40S

void fill_array(int N, int* x) {
    for (int i = 0; i < N; i++) {
        x[i] = rand() % 10;
    }
}

__device__ void warpReduceUnrolled(volatile int* s, int tid) {
    s[tid] += s[tid + 32];
    s[tid] += s[tid + 16];
    s[tid] += s[tid + 8];
    s[tid] += s[tid + 4];
    s[tid] += s[tid + 2];
    s[tid] += s[tid + 1];
}

__global__ void unrollLastWarpReduction_atomic(int *in, int *out, int n) {
    extern __shared__ int subArray[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    int val1 = (gid < n) ? in[gid] : 0;
    int val2 = (gid + blockDim.x < n) ? in[gid + blockDim.x] : 0;

    subArray[tid] = val1 + val2;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            subArray[tid] += subArray[tid + stride];
        }
        __syncthreads();
    }

    if (tid < 32) warpReduceUnrolled(subArray, tid);

    if (tid == 0) atomicAdd(out, subArray[0]);
}

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : (1 << 28);  // 2^28 = ~268M
    cout << "Summing " << N << " integers (Last-Warp Unroll + Atomic Add)\n";

    // Allocate and initialize memory
    int *in, *out_gpu;
    cudaMallocManaged(&in, N * sizeof(int));
    cudaMallocManaged(&out_gpu, sizeof(int));
    *out_gpu = 0;
    fill_array(N, in);

    const int THREAD_COUNT = BLOCK_SIZE;
    const int BLOCK_COUNT = (N + THREAD_COUNT * 2 - 1) / (THREAD_COUNT * 2);

    // Launch kernel and time it
    auto start_gpu = chrono::high_resolution_clock::now();
    unrollLastWarpReduction_atomic<<<BLOCK_COUNT, THREAD_COUNT, THREAD_COUNT * sizeof(int)>>>(in, out_gpu, N);
    cudaDeviceSynchronize();
    auto end_gpu = chrono::high_resolution_clock::now();
    chrono::duration<double> gpu_time = end_gpu - start_gpu;

    // Output result
    cout << "GPU Array sum result : " << *out_gpu << endl;
    cout << "GPU Array sum time   : " << gpu_time.count() << " seconds\n";

    // Cleanup
    cudaFree(in);
    cudaFree(out_gpu);

    return 0;
}
