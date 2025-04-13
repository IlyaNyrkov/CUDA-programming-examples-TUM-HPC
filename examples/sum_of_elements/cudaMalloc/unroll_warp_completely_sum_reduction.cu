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
    extern __shared__ int subArr[];
    
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * (blockDim.x*2) + threadIdx.x;
    subArr[tid] = in[gid] + in[gid + blockDim.x];
    __syncthreads();

    if (blockSize >= 512) { if (tid < 256) subArray[tid] += subArray[tid + 256]; __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) subArray[tid] += subArray[tid + 128]; __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) subArray[tid] += subArray[tid + 64]; __syncthreads(); }

    if (tid < 32) warpReduceTemplate<blockSize>(subArray, tid);

    if (tid == 0) {
        partialSums[blockIdx.x] = subArray[0];
    }
}

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : (1 << 26);
    bool verify = (argc > 2 && std::string(argv[2]) == "--verify");
    int blockSize = (argc > 3) ? atoi(argv[3]) : DEFAULT_BLOCK_SIZE;

    if (N <= 0 || (blockSize != 128 && blockSize != 256 && blockSize != 512)) {
        cerr << "Usage: ./prog [N] [--verify] [blockSize: 128|256|512]\n";
        return 1;
    }

    cout << "Summing " << N << " integers using Warp-unrolled reduction (blockSize = " << blockSize << ")\n";

    int* h_input = (int*)malloc(N * sizeof(int));
    fill_array(N, h_input);

    int *d_input, *d_partialSums;
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    const int THREAD_COUNT = blockSize;
    const int BLOCK_COUNT = (N + THREAD_COUNT * 2 - 1) / (THREAD_COUNT * 2);
    cudaMalloc(&d_partialSums, BLOCK_COUNT * sizeof(int));

    auto start_gpu = chrono::high_resolution_clock::now();

    switch (blockSize) {
        case 128:
            unrollWarpCompletelyReduction<128><<<BLOCK_COUNT, 128, 128 * sizeof(int)>>>(d_input, d_partialSums, N);
            break;
        case 256:
            unrollWarpCompletelyReduction<256><<<BLOCK_COUNT, 256, 256 * sizeof(int)>>>(d_input, d_partialSums, N);
            break;
        case 512:
            unrollWarpCompletelyReduction<512><<<BLOCK_COUNT, 512, 512 * sizeof(int)>>>(d_input, d_partialSums, N);
            break;
    }

    cudaDeviceSynchronize();
    auto end_gpu = chrono::high_resolution_clock::now();

    int* h_partialSums = (int*)malloc(BLOCK_COUNT * sizeof(int));
    cudaMemcpy(h_partialSums, d_partialSums, BLOCK_COUNT * sizeof(int), cudaMemcpyDeviceToHost);

    long long final_sum = 0;
    for (int i = 0; i < BLOCK_COUNT; i++) final_sum += h_partialSums[i];

    chrono::duration<double> gpu_time = end_gpu - start_gpu;
    cout << "GPU Array sum result      : " << final_sum << endl;
    cout << "GPU total reduction time  : " << gpu_time.count() << " seconds\n";

    if (verify) {
        auto start_cpu = chrono::high_resolution_clock::now();
        long long cpu_sum = accumulate(h_input, h_input + N, 0LL);
        auto end_cpu = chrono::high_resolution_clock::now();
        chrono::duration<double> cpu_time = end_cpu - start_cpu;

        cout << "CPU Accumulate sum        : " << cpu_sum << endl;
        cout << "CPU sum time              : " << cpu_time.count() << " seconds\n";

        if (cpu_sum != final_sum)
            cout << "⚠️ MISMATCH: CPU and GPU results differ!\n";
        else
            cout << "✅ CPU and GPU results match.\n";
    }

    cudaFree(d_input);
    cudaFree(d_partialSums);
    free(h_input);
    free(h_partialSums);
    return 0;
}