#include <iostream>
#include <cuda_fp16.h> // Include CUDA half-precision support
#include <cuda_runtime.h>
#include <cub/cub.cuh> // Include CUB library
#include <chrono>
#include <numeric>
#include <vector>
#include <tuple>
using namespace std;

template <unsigned int blockSize>
__global__ void warpShuffleReduction(__half *in, __half *partialSums, int n) {
    __shared__ __half warpResults[blockSize / 32]; // Shared memory for warp-level results

    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    __half sum = __float2half(0.0f);

    // Each thread loads one element
    if (gid < n) {
        sum = in[gid];
    }

    // Perform warp-level reduction using shuffle instructions
    for (int offset = 16; offset > 0; offset /= 2) {
        sum = __hadd(sum, __shfl_down_sync(0xffffffff, sum, offset));
    }

    // Store the result of each warp in shared memory
    if (tid % 32 == 0) {
        warpResults[tid / 32] = sum;
    }

    __syncthreads();

    // Perform block-level reduction on warp results using a single warp
    if (tid < 32) {
        sum = (tid < blockDim.x / 32) ? warpResults[tid] : __float2half(0.0f);
        for (int offset = 16; offset > 0; offset /= 2) {
            sum = __hadd(sum, __shfl_down_sync(0xffffffff, sum, offset));
        }
        if (tid == 0) {
            partialSums[blockIdx.x] = sum;
        }
    }
}

template <unsigned int blockSize>
__global__ void warpShuffleReductionVectorized(__half *in, __half *partialSums, int n) {
    __shared__ __half warpResults[blockSize / 32]; // Shared memory for warp-level results

    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    __half sum = __float2half(0.0f);

    // Calculate the base index for this thread
    int baseIndex = gid * 2;

    // Load and sum 8 __half2 vectors (16 half values)
    for (int i = 0; i < 8; i++) {
        int index = baseIndex + i * blockDim.x * 2; // Step by blockDim.x * 2 for each __half2
        if (index < n) {
            __half2 data = reinterpret_cast<__half2*>(in)[index / 2];
            sum = __hadd(sum, __low2half(data));
            sum = __hadd(sum, __high2half(data));
        }
    }

    // Perform warp-level reduction using shuffle instructions
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum = __hadd(sum, __shfl_down_sync(0xffffffff, sum, offset));
    }

    // Store the result of each warp in shared memory
    if (tid % 32 == 0) {
        warpResults[tid / 32] = sum;
    }

    __syncthreads();

    // Perform block-level reduction on warp results using a single warp
    if (tid < 32) {
        sum = (tid < blockDim.x / 32) ? warpResults[tid] : __float2half(0.0f);
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum = __hadd(sum, __shfl_down_sync(0xffffffff, sum, offset));
        }
        if (tid == 0) {
            partialSums[blockIdx.x] = sum; // Write the final block sum to global memory
        }
    }
}

// Kernel to sum the partial results from all blocks
__global__ void finalReduction(__half *partialSums, __half *result, int n) {
    __half sum = __float2half(0.0f);

    // Each thread processes 4 elements at a time using __half2
    int tid = threadIdx.x;
    int numThreads = blockDim.x;
    int numVectors = n / 2; // Number of __half2 vectors
    int remainder = n % 2; // Remaining elements

    // Process 2 elements at a time using __half2
    for (int i = tid; i < numVectors; i += numThreads) {
        __half2 data = reinterpret_cast<__half2*>(partialSums)[i];
        sum = __hadd(sum, __low2half(data));
        sum = __hadd(sum, __high2half(data));
    }

    // Handle remaining elements (if n is not a multiple of 2)
    if (tid < remainder) {
        sum = __hadd(sum, partialSums[numVectors * 2 + tid]);
    }

    // Perform warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        sum = __hadd(sum, __shfl_down_sync(0xffffffff, sum, offset));
    }

    // Store the final result
    if (tid == 0) {
        atomicAdd(reinterpret_cast<float*>(result), __half2float(sum));
    }
}

template <typename Kernel>
std::tuple<float, float> benchmarkKernel(Kernel kernel, __half *dev_input_data, __half *dev_partial_sums, __half *dev_result, int n, int blockSize, int factor) {
    // Calculate the number of blocks
    int numBlocks = (n + blockSize * factor - 1) / (blockSize * factor);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timing
    cudaEventRecord(start);
    kernel<<<numBlocks, blockSize>>>(dev_input_data, dev_partial_sums, n);
    finalReduction<<<1, 32>>>(dev_partial_sums, dev_result, numBlocks);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float duration = 0.0f;
    cudaEventElapsedTime(&duration, start, stop);

    // Retrieve the result from the device
    float result;
    cudaMemcpy(&result, dev_result, sizeof(float), cudaMemcpyDeviceToHost);

    // Reset device result memory
    cudaMemset(dev_result, 0, sizeof(__half));

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return {result, duration};
}

int main() {
    int n = 1 << 25; // 4M elements
    size_t bytes = n * sizeof(__half);

    // Host/CPU arrays
    __half *host_input_data = new __half[n];

    // Device/GPU arrays
    __half *dev_input_data, *dev_partial_sums, *dev_result;

    // Initialize data
    srand(42); // Fixed seed
    for (int i = 0; i < n; i++) {
        host_input_data[i] = __float2half(static_cast<float>(rand()) / RAND_MAX); // Random floats between 0 and 1
    }

    // Allocate memory on GPU
    cudaMalloc(&dev_input_data, bytes);
    cudaMalloc(&dev_partial_sums, ((n + 1023) / 1024) * sizeof(__half));
    cudaMalloc(&dev_result, sizeof(__half));
    cudaMemset(dev_result, 0, sizeof(__half));

    // Copy data to GPU
    cudaMemcpy(dev_input_data, host_input_data, bytes, cudaMemcpyHostToDevice);

    int blockSize = 1024; // Threads per block

    // Map-like structure to store results and durations
    std::vector<std::tuple<std::string, float, float>> results;

    // Perform CPU reduction
    auto cpu_start = std::chrono::high_resolution_clock::now();
    float cpu_result = 0.0f;
    for (int i = 0; i < n; i++) {
        cpu_result += __half2float(host_input_data[i]);
    }
    auto cpu_stop = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_stop - cpu_start).count() / 1000.0f;
    results.emplace_back("CPU", cpu_result, cpu_duration);

    // Benchmark different kernels
    {
        auto [result, duration] = benchmarkKernel(warpShuffleReduction<1024>, dev_input_data, dev_partial_sums, dev_result, n, blockSize, 1);
        results.emplace_back("Restored", result, duration);
    }
    {
        auto [result, duration] = benchmarkKernel(warpShuffleReductionVectorized<1024>, dev_input_data, dev_partial_sums, dev_result, n, blockSize, 8);
        results.emplace_back("Vectorized", result, duration);
    }

    // CUB reduction
    float *host_cub_input_float = new float[n]; // Allocate a float array on the host
    float *dev_cub_input, *dev_cub_result;
    
    // Allocate memory on the device
    cudaMalloc(&dev_cub_input, n * sizeof(float));
    cudaMalloc(&dev_cub_result, sizeof(float));
    cudaMemset(dev_cub_result, 0, sizeof(float));
    
    // Convert __half input to float for CUB
    for (int i = 0; i < n; i++) {
        host_cub_input_float[i] = __half2float(host_input_data[i]); // Convert __half to float
    }
    
    // Copy the converted float array to the device
    cudaMemcpy(dev_cub_input, host_cub_input_float, n * sizeof(float), cudaMemcpyHostToDevice);
    
    // CUB temporary storage
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    // Determine temporary device storage requirements
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, dev_cub_input, dev_cub_result, n);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    // Perform CUB reduction
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, dev_cub_input, dev_cub_result, n);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float cub_duration = 0.0f;
    cudaEventElapsedTime(&cub_duration, start, stop);
    
    // Retrieve the result from the device
    float cub_result;
    cudaMemcpy(&cub_result, dev_cub_result, sizeof(float), cudaMemcpyDeviceToHost);
    
    // Add the result to the results vector
    results.emplace_back("CUB", cub_result, cub_duration);
    

    // Display results and verify correctness
    for (const auto &[technique, result, duration] : results) {
        std::cout << "Technique: " << technique << ", Result: " << result
                  << ", Time: " << duration << " ms" << std::endl;
    }

    // Free memory
    cudaFree(dev_input_data);
    cudaFree(dev_partial_sums);
    cudaFree(dev_result);
    cudaFree(dev_cub_input);
    cudaFree(dev_cub_result);
    cudaFree(d_temp_storage);
    delete[] host_input_data;

    return 0;
}