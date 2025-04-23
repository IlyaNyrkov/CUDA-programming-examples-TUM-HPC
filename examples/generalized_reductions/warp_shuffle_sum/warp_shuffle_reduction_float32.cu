#include <iostream>
#include <cuda_runtime.h>
#include <cub/cub.cuh> // Include CUB library
#include <chrono>
#include <numeric>
#include <vector>
#include <tuple>
using namespace std;

template <unsigned int blockSize>
__global__ void warpShuffleReduction(float *in, float *partialSums, int n) {
    __shared__ float warpResults[32]; // Shared memory for warp-level results (32 warps per block)

    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    // Each thread loads one element
    if (gid < n) {
        sum = in[gid];
    }

    // Perform warp-level reduction using shuffle instructions
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Store the result of each warp in shared memory
    if (tid % 32 == 0) {
        warpResults[tid / 32] = sum;
    }

    __syncthreads();

    // Perform block-level reduction on warp results using a single warp
    if (tid < 32) {
        sum = (tid < blockDim.x / 32) ? warpResults[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (tid == 0) {
            partialSums[blockIdx.x] = sum;
        }
    }
}

template <unsigned int blockSize>
__global__ void warpShuffleReductionVectorized(float *in, float *partialSums, int n) {
    __shared__ float warpResults[32]; // Shared memory for warp-level results (32 warps per block)

    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    float sum = 0.0f;

    // Calculate the base index for this thread
    int baseIndex = gid * 4;

    // Load and sum 8 float4 vectors (32 floats)
    for (int i = 0; i < 8; i++) {
        int index = baseIndex + i * blockDim.x * 4; // Step by blockDim.x * 4 for each float4
        if (index < n) {
            float4 data = reinterpret_cast<float4*>(in)[index / 4];
            sum += data.x;
            sum += data.y;
            sum += data.z;
            sum += data.w;
        }
    }

    // Perform warp-level reduction using shuffle instructions
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Store the result of each warp in shared memory
    if (tid % 32 == 0) {
        warpResults[tid / 32] = sum;
    }

    __syncthreads();

    // Perform block-level reduction on warp results using a single warp
    if (tid < 32) {
        sum = (tid < blockDim.x / 32) ? warpResults[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (tid == 0) {
            partialSums[blockIdx.x] = sum; // Write the final block sum to global memory
        }
    }
}

// Kernel to sum the partial results from all blocks
__global__ void finalReduction(float *partialSums, float *result, int n) {
    float sum = 0.0f;

    // Each thread processes 4 elements at a time using float4
    int tid = threadIdx.x;
    int numThreads = blockDim.x;
    int numVectors = n / 4; // Number of float4 vectors
    int remainder = n % 4; // Remaining elements

    // Process 4 elements at a time using float4
    for (int i = tid; i < numVectors; i += numThreads) {
        float4 data = reinterpret_cast<float4*>(partialSums)[i];
        sum += data.x + data.y + data.z + data.w;
    }

    // Handle remaining elements (if n is not a multiple of 4)
    if (tid < remainder) {
        sum += partialSums[numVectors * 4 + tid];
    }

    // Perform warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Store the final result
    if (tid == 0) {
        atomicAdd(result, sum);
    }
}

template <typename Kernel>
std::tuple<float, float> benchmarkKernel(Kernel kernel, float *dev_input_data, float *dev_partial_sums, float *dev_result, int n, int blockSize, int factor) {
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
    cudaMemset(dev_result, 0, sizeof(float));

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return {result, duration};
}

void displayAndVerifyResults(const std::vector<std::tuple<std::string, float, float>> &results, float cpu_result) {
    bool all_match = true;

    for (const auto &[technique, result, duration] : results) {
        std::cout << "Technique: " << technique << ", Result: " << result
                  << ", Time: " << duration << " ms" << std::endl;

        if (fabs(result - cpu_result) > 1e-5) { // Allow small floating-point error
            all_match = false;
            std::cout << "\033[31mMismatch in " << technique << ": " << result
                      << " (Expected: " << cpu_result << ")\033[0m\n";
        }
    }

    if (all_match) {
        std::cout << "\033[32mVerification successful: All results match.\033[0m\n";
    } else {
        std::cout << "\033[31mVerification failed: Some results do not match.\033[0m\n";
    }
}

int main() {
    int n = 1 << 25; // 4M elements
    size_t bytes = n * sizeof(float);

    // Host/CPU arrays
    float *host_input_data = new float[n];

    // Device/GPU arrays
    float *dev_input_data, *dev_partial_sums, *dev_result;

    // Initialize data
    srand(42); // Fixed seed
    for (int i = 0; i < n; i++) {
        host_input_data[i] = static_cast<float>(rand()) / RAND_MAX; // Random floats between 0 and 1
    }

    // Allocate memory on GPU
    cudaMalloc(&dev_input_data, bytes);
    cudaMalloc(&dev_partial_sums, ((n + 1023) / 1024) * sizeof(float));
    cudaMalloc(&dev_result, sizeof(float));
    cudaMemset(dev_result, 0, sizeof(float));

    // Copy data to GPU
    cudaMemcpy(dev_input_data, host_input_data, bytes, cudaMemcpyHostToDevice);

    int blockSize = 1024; // Threads per block

    // Map-like structure to store results and durations
    std::vector<std::tuple<std::string, float, float>> results;

    // Perform CPU reduction
    auto cpu_start = std::chrono::high_resolution_clock::now();
    float cpu_result = std::accumulate(host_input_data, host_input_data + n, 0.0f);
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
    float *dev_cub_result;
    cudaMalloc(&dev_cub_result, sizeof(float));
    cudaMemset(dev_cub_result, 0, sizeof(float)); // Initialize to 0

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Determine temporary device storage requirements
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, dev_input_data, dev_cub_result, n);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, dev_input_data, dev_cub_result, n);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float cub_duration = 0.0f;
    cudaEventElapsedTime(&cub_duration, start, stop);
    float cub_result;
    cudaMemcpy(&cub_result, dev_cub_result, sizeof(float), cudaMemcpyDeviceToHost);
    results.emplace_back("CUB", cub_result, cub_duration);

    // Display results and verify correctness
    displayAndVerifyResults(results, cpu_result);

    // Free memory
    cudaFree(dev_input_data);
    cudaFree(dev_partial_sums);
    cudaFree(dev_result);
    cudaFree(dev_cub_result);
    cudaFree(d_temp_storage);
    delete[] host_input_data;

    return 0;
}