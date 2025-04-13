#include <stdio.h>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <numeric>


using namespace std;

#define BLOCK_SIZE 256  // L40S/H100 optimal: 256–512

void fill_array(int N, int* x) {
    for (int i = 0; i < N; i++) {
        x[i] = rand() % 100;
    }
}

__global__ void firstAddGlobalLoadReduction(int *in, int *partialSums, int n) {
    extern __shared__ int subArr[];
    
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * (blockDim.x*2) + threadIdx.x;
    subArr[tid] = in[gid] + in[gid + blockDim.x];
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
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
    int N = (argc > 1) ? atoi(argv[1]) : (1 << 26);  // Default: 64M ints
    bool verify_cpu = (argc > 2 && string(argv[2]) == "--verify");

    cout << "Summing " << N << " integers using First-Add-Global-Load (GPU-only + CPU final sum)\n";

    // --- Host allocation
    int *h_input = (int*)malloc(N * sizeof(int));
    fill_array(N, h_input);

    // --- Device allocation
    int *d_input;
    cudaMalloc(&d_input, N * sizeof(int));

    const int THREAD_COUNT = BLOCK_SIZE;
    const int BLOCK_COUNT = (N + THREAD_COUNT * 2 - 1) / (THREAD_COUNT * 2);

    int *d_partial_sums;
    cudaMalloc(&d_partial_sums, BLOCK_COUNT * sizeof(int));

    // --- Copy input data to device
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    // --- Launch kernel
    auto start_gpu = chrono::high_resolution_clock::now();
    firstAddGlobalLoadReduction<<<BLOCK_COUNT, THREAD_COUNT, THREAD_COUNT * sizeof(int)>>>(d_input, d_partial_sums, N);
    cudaDeviceSynchronize();

    // --- Copy partial results back
    int *h_partial_sums = (int*)malloc(BLOCK_COUNT * sizeof(int));
    cudaMemcpy(h_partial_sums, d_partial_sums, BLOCK_COUNT * sizeof(int), cudaMemcpyDeviceToHost);

    long long final_sum = 0;
    for (int i = 0; i < BLOCK_COUNT; i++) {
        final_sum += h_partial_sums[i];
    }
    auto end_gpu = chrono::high_resolution_clock::now();
    chrono::duration<double> gpu_time = end_gpu - start_gpu;

    // --- Output results
    cout << "GPU Array sum result       : " << final_sum << endl;
    cout << "GPU total reduction time   : " << gpu_time.count() << " seconds\n";

    if (verify_cpu) {
        auto start_cpu = chrono::high_resolution_clock::now();
        long long cpu_sum = accumulate(h_input, h_input + N, 0LL);
        auto end_cpu = chrono::high_resolution_clock::now();
        chrono::duration<double> cpu_time = end_cpu - start_cpu;

        cout << "CPU std::accumulate result : " << cpu_sum << endl;
        cout << "CPU time                   : " << cpu_time.count() << " seconds\n";

        if (cpu_sum != final_sum) {
            cout << "⚠️ WARNING: GPU and CPU results do not match!" << endl;
        } else {
            cout << "✅ GPU result matches CPU!" << endl;
        }
    }

    // --- Cleanup
    free(h_input);
    free(h_partial_sums);
    cudaFree(d_input);
    cudaFree(d_partial_sums);

    return 0;
}