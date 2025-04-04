#include <stdio.h>
#include <time.h>
#include <iostream>
#include <chrono>

using namespace std;

#define BLOCK_SIZE 256   // Tune for your GPU (L40S handles 256–512 well)


void fill_array(int N, int* x) {
    for (int i = 0; i < N; i++) {
        x[i] = 1;
    }
}

void print_array(int N, int* x) {
    printf("printing first %d elements", N);
    for (int i = 0; i < N; i++) {
        printf("%d ", x[i]);
    }

    printf("\n");
}

void sumCPU(int *input, int *output, int n) {
    for (int i = 0; i < n; i++) {
        (*output) += input[i];
    }
}

template <unsigned int blockSize>
__device__ void warpReduceTemplate(volatile int* s, int tid) {
    if (blockSize >= 64) s[tid] += s[tid + 32];
    if (blockSize >= 32) s[tid] += s[tid + 16];
    if (blockSize >= 16) s[tid] += s[tid + 8];
    if (blockSize >= 8) s[tid] += s[tid + 4];
    if (blockSize >= 4) s[tid] += s[tid + 2];
    if (blockSize >= 2) s[tid] += s[tid + 1];
}

template <unsigned int blockSize>
__global__ void gridStrideReduction(int *in, int *out, int n) {
    extern __shared__ int subArr[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockSize * 2 + threadIdx.x;
    int gridSize = blockSize * gridDim.x * 2;
    subArr[tid] = 0;
   
    while (gid < n) { subArr[tid] += in[gid] + in[gid+blockSize]; gid += gridSize; };
    __syncthreads();
    

    // if (blockSize >= 1024) { if (tid < 512) subArr[tid] += subArr[tid + 512]; __syncthreads(); }
    if (blockSize >= 512)  { if (tid < 256) subArr[tid] += subArr[tid + 256]; __syncthreads(); }
    if (blockSize >= 256)  { if (tid < 128) subArr[tid] += subArr[tid + 128]; __syncthreads(); }
    if (blockSize >= 128)  { if (tid < 64)  subArr[tid] += subArr[tid + 64];  __syncthreads(); };

    if (tid < 32) warpReduceTemplate<blockSize>(subArr, tid);
    if (tid == 0) atomicAdd(out, subArr[0]);
}

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : (1 << 28);

    cout << "Summing " << N << " integers using CPU and GPU (Grid-stride + Warp unroll)\n";

    // Allocate unified memory
    int *input, *output_gpu;
    int output_cpu = 0;
    cudaMallocManaged(&input, N * sizeof(int));
    cudaMallocManaged(&output_gpu, sizeof(int));
    *output_gpu = 0;

    fill_array(N, input);

    // --- CPU SUM ---
    auto start_cpu = chrono::high_resolution_clock::now();
    sumCPU(input, &output_cpu, N);
    auto end_cpu = chrono::high_resolution_clock::now();
    chrono::duration<double> cpu_time = end_cpu - start_cpu;

    cout << "CPU Array sum result: " << output_cpu << endl;
    cout << "CPU Array sum time  : " << cpu_time.count() << " seconds" << endl;

    // --- GPU SUM ---
    const int THREAD_COUNT = BLOCK_SIZE;
    const int BLOCK_COUNT = (N + (THREAD_COUNT * 2 - 1)) / (THREAD_COUNT * 2); // 2 elements per thread

    *output_gpu = 0;
    auto start_gpu = chrono::high_resolution_clock::now();
    gridStrideReduction<THREAD_COUNT><<<BLOCK_COUNT, THREAD_COUNT, THREAD_COUNT * sizeof(int)>>>(input, output_gpu, N);
    cudaDeviceSynchronize();
    auto end_gpu = chrono::high_resolution_clock::now();
    chrono::duration<double> gpu_time = end_gpu - start_gpu;

    cout << "GPU Array sum result: " << *output_gpu << endl;
    cout << "GPU Array sum time  : " << gpu_time.count() << " seconds" << endl;

    // Cleanup
    cudaFree(input);
    cudaFree(output_gpu);

    return 0;
}
