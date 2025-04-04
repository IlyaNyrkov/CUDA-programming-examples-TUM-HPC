#include <stdio.h>
#include <stdlib.h>
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
    printf("printing first %d elements:\n", N);
    for (int i = 0; i < N; i++) {
        printf("%d ", x[i]);
    }
    printf("\n");
}

__global__ void naiveSum(int *input, int *output, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        atomicAdd(output, input[idx]);
    }
}

void sumCPU(int *input, int *output, int n) {
    for (int i = 0; i < n; i++) {
        (*output) += input[i];
    }
}

int main(int argc, char* argv[]) {
    // Default number of elements: 2^30
    int N = (argc > 1) ? atoi(argv[1]) : (1 << 28);
    const int max_val = 10;

    if (N <= 0) {
        cerr << "Invalid number of elements. Must be a positive integer.\n";
        return 1;
    }

    cout << "Summing up " << N << " integers...\n";

    // Allocate unified memory
    int *input, *output_gpu;
    int output_cpu = 0;
    cudaMallocManaged(&input, N * sizeof(int));
    cudaMallocManaged(&output_gpu, sizeof(int));
    *output_gpu = 0;

    // Fill and display input
    fill_array(N, max_val, input);
    print_array(min(N, 10), input);  // Print first 10 elements

    // --- CPU SUM ---
    auto start_cpu = chrono::high_resolution_clock::now();
    sumCPU(input, &output_cpu, N);
    auto end_cpu = chrono::high_resolution_clock::now();
    chrono::duration<double> cpu_time = end_cpu - start_cpu;

    cout << "CPU Array sum result: " << output_cpu << endl;
    cout << "CPU Array sum time  : " << cpu_time.count() << " seconds\n";

    // --- GPU SUM ---
    const int THREAD_COUNT = 128;
    const int BLOCK_COUNT = (N + THREAD_COUNT - 1) / THREAD_COUNT;

    *output_gpu = 0;
    auto start_gpu = chrono::high_resolution_clock::now();
    naiveSum<<<BLOCK_COUNT, THREAD_COUNT>>>(input, output_gpu, N);
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


