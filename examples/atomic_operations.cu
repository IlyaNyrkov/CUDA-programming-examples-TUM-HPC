include <stdio.h>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <climits>
using namespace std;

__device__ int result_gpu;

__global__ void atomicAddKernel(int n, int *x) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        atomicAdd(&result_gpu, x[id]);
    }
}

__global__ void atomicSubKernel(int n, int *x) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        atomicSub(&result_gpu, x[id]);
    }
}

__global__ void atomicExchKernel(int n, int *x) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        atomicExch(&result_gpu, x[id]); // Sets result_gpu to the last thread's value
    }
}

__global__ void atomicMinKernel(int n, int *x) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        atomicMin(&result_gpu, x[id]);
    }
}

__global__ void atomicMaxKernel(int n, int *x) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        atomicMax(&result_gpu, x[id]);
    }
}

void fill_array(int N, int max_val, int *x) {
    for (int i = 0; i < N; i++) {
        x[i] = rand() % (max_val + 1);
    }
}

void print_array(int N, int *x) {
    for (int i = 0; i < N; i++) {
        std::cout << x[i] << " "; 
    }
}

// Helper function to execute a kernel and print results
template <typename KernelFunc>
void executeAtomicOperation(KernelFunc kernel, const char *op_name, int N, int *x, int initial_val) {
    const int THREAD_COUNT = 256;
    const int BLOCK_COUNT = (N + 255) / THREAD_COUNT;

    // Initialize the result variable on the GPU
    cudaMemcpyToSymbol(result_gpu, &initial_val, sizeof(int));

    // Launch kernel
    auto start_gpu = chrono::high_resolution_clock::now();
    kernel<<<BLOCK_COUNT, THREAD_COUNT>>>(N, x);
    cudaDeviceSynchronize();
    auto end_gpu = chrono::high_resolution_clock::now();

    // Copy result back to CPU
    int result_cpu;
    cudaMemcpyFromSymbol(&result_cpu, result_gpu, sizeof(int));

    // Calculate time
    chrono::duration<double> gpu_time = end_gpu - start_gpu;

    // Print results
    cout << "\nOperation: " << op_name << endl;
    cout << "Result: " << result_cpu << endl;
    cout << "Execution Time: " << gpu_time.count() << " seconds" << endl;
}

// Main function
int main() {
    int N = 1 << 20; // 2^20 elements (large enough for performance demonstration)
    int *x;

    // Allocate unified memory
    cudaMallocManaged(&x, N * sizeof(int));
    fill_array(N, 30, x);
    print_array(N, x);

    // Execute atomic operations
    executeAtomicOperation(atomicAddKernel, "atomicAdd", N, x, 0);        // Start with 0
    executeAtomicOperation(atomicSubKernel, "atomicSub", N, x, 0);        // Start with 0
    executeAtomicOperation(atomicExchKernel, "atomicExch", N, x, 0);      // Start with 0
    executeAtomicOperation(atomicMinKernel, "atomicMin", N, x, INT_MAX);  // Start with INT_MAX
    executeAtomicOperation(atomicMaxKernel, "atomicMax", N, x, INT_MIN);  // Start with INT_MIN

    // Free memory
    cudaFree(x);

    return 0;
}