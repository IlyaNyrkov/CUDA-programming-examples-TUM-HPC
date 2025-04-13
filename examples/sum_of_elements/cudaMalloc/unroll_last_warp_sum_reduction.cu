#include <stdio.h>
#include <iostream>
#include <chrono>
#include <numeric>


using namespace std;

#define BLOCK_SIZE 256  // L40S/H100 optimal: 256–512

void fill_array(int N, int* x) {
    for (int i = 0; i < N; i++) {
        x[i] = rand() % 100;
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

__global__ void unrollLastWarpReduction(int *in, int *partialSums) {
    extern __shared__ int subArr[];
    
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * (blockDim.x*2) + threadIdx.x;
    subArr[tid] = in[gid] + in[gid + blockDim.x];
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            subArr[tid] += subArr[tid + stride];
        }
        __syncthreads();
    }

    if (tid < 32) warpReduceUnrolled(subArr, tid);

    if (tid == 0) {
        partialSums[blockIdx.x] = subArr[0];
    }
}

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : (1 << 26);
    bool verify = (argc > 2 && strcmp(argv[2], "--verify") == 0);

    cout << "Summing " << N << " integers (Last-Warp Unroll + CPU final sum)\n";

    // Allocate host memory
    int *h_input = (int*) malloc(N * sizeof(int));
    fill_array(N, h_input);

    // Allocate device memory
    int *d_input;
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    const int THREAD_COUNT = BLOCK_SIZE;
    const int BLOCK_COUNT = (N + THREAD_COUNT * 2 - 1) / (THREAD_COUNT * 2);

    int *d_partialSums;
    int *h_partialSums = (int*) malloc(BLOCK_COUNT * sizeof(int));
    cudaMalloc(&d_partialSums, BLOCK_COUNT * sizeof(int));

    // GPU timing
    auto start_gpu = chrono::high_resolution_clock::now();
    unrollLastWarpReduction<<<BLOCK_COUNT, THREAD_COUNT, THREAD_COUNT * sizeof(int)>>>(d_input, d_partialSums);
    cudaDeviceSynchronize();
    auto end_gpu = chrono::high_resolution_clock::now();

    // Copy result back and sum on CPU
    cudaMemcpy(h_partialSums, d_partialSums, BLOCK_COUNT * sizeof(int), cudaMemcpyDeviceToHost);
    long long gpu_result = 0;
    for (int i = 0; i < BLOCK_COUNT; i++) {
        gpu_result += h_partialSums[i];
    }

    chrono::duration<double> gpu_time = end_gpu - start_gpu;

    // Output
    cout << "GPU Array sum result : " << gpu_result << endl;
    cout << "GPU Array sum time   : " << gpu_time.count() << " seconds\n";

    // Optional verification
    if (verify) {
        long long cpu_result = std::accumulate(h_input, h_input + N, 0LL);
        cout << "CPU Array sum result : " << cpu_result << endl;
        if (cpu_result != gpu_result) {
            cout << "[WARNING] GPU result does not match CPU result!\n";
        } else {
            cout << "[OK] GPU and CPU results match.\n";
        }
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_partialSums);
    free(h_input);
    free(h_partialSums);

    return 0;
}