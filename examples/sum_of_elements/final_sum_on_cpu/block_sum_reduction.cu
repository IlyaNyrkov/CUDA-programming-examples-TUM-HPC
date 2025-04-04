#include <stdio.h>
#include <time.h>
#include <iostream>
#include <chrono>

using namespace std;

#define BLOCK_SIZE 256   // Tune for your GPU (L40S handles 256–512 well)

void fill_array(int N, int max_val, int* x) {
    srand(time(0));
    for (int i = 0; i < N; i++) {
        x[i] = rand() % (max_val + 1);
    }
}

void print_array(int N, int* x) {
    printf("printing first %d elements: ", N);
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
        out[blockIdx.x] = subArray[0]; // Store per-block result
    }
}

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : (1 << 28);
    const int max = 10;

    cout << "Summing " << N << " integers using CPU and GPU (Block-wise reduction)\n";

    // Allocate and fill input
    int *input;
    cudaMallocManaged(&input, N * sizeof(int));
    fill_array(N, max, input);
    print_array(min(N, 10), input);

    // ------------------ CPU SUM ------------------
    int output_cpu = 0;
    auto start_cpu = chrono::high_resolution_clock::now();
    sumCPU(input, &output_cpu, N);
    auto end_cpu = chrono::high_resolution_clock::now();
    chrono::duration<double> cpu_time = end_cpu - start_cpu;

    cout << "CPU Array sum result: " << output_cpu << endl;
    cout << "CPU Array sum time  : " << cpu_time.count() << " seconds\n";

    // ------------------ GPU SUM ------------------
    const int THREAD_COUNT = BLOCK_SIZE;
    const int BLOCK_COUNT = (N + THREAD_COUNT * 2 - 1) / (THREAD_COUNT * 2);

    int *partial_sums;
    cudaMallocManaged(&partial_sums, BLOCK_COUNT * sizeof(int));

    auto start_gpu = chrono::high_resolution_clock::now();
    blockSumReduction<<<BLOCK_COUNT, THREAD_COUNT, THREAD_COUNT * sizeof(int)>>>(input, partial_sums, N);
    cudaDeviceSynchronize();

    // Final reduction on CPU
    long long final_sum = 0;
    for (int i = 0; i < BLOCK_COUNT; i++) {
        final_sum += partial_sums[i];
    }
    auto end_gpu = chrono::high_resolution_clock::now();
    chrono::duration<double> gpu_time = end_gpu - start_gpu;

    cout << "GPU Array sum result: " << final_sum << endl;
    cout << "GPU total reduction time: " << gpu_time.count() << " seconds\n";

    // Cleanup
    cudaFree(input);
    cudaFree(partial_sums);

    return 0;
}
