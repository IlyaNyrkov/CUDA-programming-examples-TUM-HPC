#include <stdio.h>
#include <time.h>
#include <iostream>
#include <chrono>

using namespace std;

void fill_array(int N, int max_val, int* x) {
    srand(time(0));
    for (int i = 0; i < N; i++) {
        x[i] = rand() % (max_val + 1);
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

int main() {
    const int N = 1 << 10;
    const int max = 100;

    int* input, *output_cpu, *output_gpu;
    cudaMallocManaged(&input, N*sizeof(int));
    fill_array(N, 100, input);
    print_array(10, input);

    auto start_cpu = chrono::high_resolution_clock::now();
    sumCPU(input, output_cpu, N);
    auto end_cpu = chrono::high_resolution_clock::now();

    chrono::duration<double> cpu_time = end_cpu - start_cpu;

    cout << "CPU Array sum time: " << cpu_time.count() << " seconds" << endl;
    cout << "CPU Array sum result: " << *output_cpu << endl; 

    const int THREAD_COUNT = 128;
    const int BLOCK_COUNT = (N + (THREAD_COUNT - 1)) / THREAD_COUNT;

    auto start_gpu = chrono::high_resolution_clock::now();
    firstAddGlobalLoadReduction<<<BLOCK_COUNT, THREAD_COUNT, THREAD_COUNT>>>(input, output_gpu, N);
    cudaDeviceSynchronize();
    auto end_gpu = chrono::high_resolution_clock::now();

    chrono::duration<double> gpu_time = end_gpu - start_gpu;

    cout << "GPU Array sum time: " << gpu_time.count() << " seconds" << endl;
    cout << "GPU Array sum result: " << *output_gpu << endl; 


}