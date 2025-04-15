#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include <iostream>
#include<cuda_runtime.h>
#include <chrono>
#include <numeric> 
using namespace std;

// Adding this function to help with unrolling and adding the Template
template <unsigned int blockSize>
__device__ void warpReduce(volatile int* sdata, unsigned int tid){
    if(blockSize >= 64) sdata[tid] += sdata[tid + 32];
    int val = sdata[tid];
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    sdata[tid] = val;
}

// REDUCTION 6 – Multiple Adds / Threads
template <int blockSize>
__global__ void reduce6(int *g_in_data, int *g_out_data, unsigned int n){
    extern __shared__ int sdata[];  // stored in the shared memory

    // Each thread loading one element from global onto shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockDim.x * 2 * gridDim.x;
    sdata[tid] = 0;

    while(i < n) { 
      sdata[tid] += g_in_data[i] + g_in_data[i + blockSize]; 
      i += gridSize; 
    }
    __syncthreads();

    // Perform reductions in steps, reducing thread synchronization
    if (blockSize >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
    }

    if (tid < 32) warpReduce<blockSize>(sdata, tid);

    if (tid == 0){
        g_out_data[blockIdx.x] = sdata[0];
    }

}


int main() {
    int n = 1 << 25; 
    size_t bytes = n * sizeof(int);

    // Host/CPU arrays
    int *host_input_data = new int[n];
    int *host_output_data = new int[(n + 255) / 256]; // enough space for partial sums

    // Device/GPU arrays
    int *dev_input_data, *dev_output_data;

    // Init data
    srand(42);
    for (int i = 0; i < n; i++) {
        host_input_data[i] = rand() % 100;
    }

    // Allocate device memory
    cudaMalloc(&dev_input_data, bytes);
    cudaMalloc(&dev_output_data, (n + 255) / 256 * sizeof(int));
    cudaMemcpy(dev_input_data, host_input_data, bytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int num_blocks = (n + (2 * blockSize) - 1) / (2 * blockSize);

    // Create CUDA events
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    // Start event recording
    cudaEventRecord(startEvent, 0);
    reduce6<256><<<num_blocks, 256, 256 * sizeof(int)>>>(dev_input_data, dev_output_data, n);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    // Measure elapsed time
    float elapsedMs = 0.0f;
    cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent); // in milliseconds

    // Copy results back
    cudaMemcpy(host_output_data, dev_output_data, (n + 255) / 256 * sizeof(int), cudaMemcpyDeviceToHost);

    int finalResult = host_output_data[0];
    for (int i = 1; i < (n + 255) / 256; ++i) {
        finalResult += host_output_data[i];
    }

    // Verification with CPU
    int cpuResult = std::accumulate(host_input_data, host_input_data + n, 0);
    if (cpuResult == finalResult) {
        std::cout << "\033[32mVerification successful: GPU result matches CPU result.\n";
    } else {
        std::cout << "\033[31mVerification failed: GPU result (" << finalResult << ") != CPU result (" << cpuResult << ").\n";
    }
    std::cout << "\033[0m"; // Reset text color

    std::cout << "GPU Time (CUDA Events): " << elapsedMs << " ms\n";
    double bandwidth = (elapsedMs > 0) ? (bytes / elapsedMs / 1e6) : 0;
    std::cout << "Effective Bandwidth: " << bandwidth << " GB/s\n";

    // Also compare to Thrust (optional)
    thrust::device_ptr<int> dev_ptr(dev_input_data);
    cudaEventRecord(startEvent, 0);
    int thrust_result = thrust::reduce(thrust::device, dev_ptr, dev_ptr + n, 0, thrust::plus<int>());
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent);

    std::cout << "Thrust Result: " << thrust_result << "\n";
    std::cout << "Thrust Time: " << elapsedMs << " ms\n";

    // Cleanup
    cudaFree(dev_input_data);
    cudaFree(dev_output_data);
    delete[] host_input_data;
    delete[] host_output_data;
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}
