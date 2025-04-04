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

__global__ void firstAddGlobalLoadReduction(int *in, int *out, int n) {
    extern __shared__ int subArr[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    int val1 = 0;
    
    if (gid < n) {
        val1 = in[gid];
    }

    int val2 = 0;

    if (gid + blockDim.x < n) {
        val2 = in[gid + blockDim.x];
    }

    subArr[tid] = val1 + val2;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            subArr[tid] += subArr[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out, subArr[0]);
    }
}

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : (1 << 28);

    cout << "Summing " << N << " integers using CPU and GPU (First Add During Global Load)\n";

    // Allocate unified memory
    int *input, *output_gpu;
    int output_cpu = 0;
    cudaMallocManaged(&input, N * sizeof(int));
    cudaMallocManaged(&output_gpu, sizeof(int));
    *output_gpu = 0;

    // Fill array
    fill_array(N, input);

    // --- CPU SUM ---
    auto start_cpu = chrono::high_resolution_clock::now();
    sumCPU(input, &output_cpu, N);
    auto end_cpu = chrono::high_resolution_clock::now();
    chrono::duration<double> cpu_time = end_cpu - start_cpu;

    cout << "CPU Array sum result: " << output_cpu << endl;
    cout << "CPU Array sum time  : " << cpu_time.count() << " seconds\n";

    // --- GPU SUM ---
    const int THREAD_COUNT = BLOCK_SIZE;
    const int BLOCK_COUNT = (N + THREAD_COUNT * 2 - 1) / (THREAD_COUNT * 2);  // 2 elements per thread

    *output_gpu = 0;
    auto start_gpu = chrono::high_resolution_clock::now();
    firstAddGlobalLoadReduction<<<BLOCK_COUNT, THREAD_COUNT, THREAD_COUNT * sizeof(int)>>>(input, output_gpu, N);
    cudaDeviceSynchronize();
    auto end_gpu = chrono::high_resolution_clock::now();
    chrono::duration<double> gpu_time = end_gpu - start_gpu;

    cout << "GPU Array sum result: " << *output_gpu << endl;
    cout << "GPU Array sum time  : " << gpu_time.count() << " seconds\n";

    // Cleanup
    cudaFree(input);
    cudaFree(output_gpu);

    return 0;
}
