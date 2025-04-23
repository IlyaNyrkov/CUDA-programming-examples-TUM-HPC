#include <iostream>
#include <cuda_runtime.h>
#include <cub/cub.cuh> // Include CUB library
#include <chrono>
#include <numeric>
using namespace std;

template <unsigned int blockSize>
__global__ void warpShuffleReduction(int *in, int *partialSums, int n) {
    __shared__ int warpResults[32]; // Shared memory for warp-level results (32 warps per block)

    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    int sum = 0;

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
        sum = (tid < blockDim.x / 32) ? warpResults[tid] : 0;
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (tid == 0) {
            partialSums[blockIdx.x] = sum;
        }
    }
}


template <unsigned int blockSize>
__global__ void warpShuffleReductionDoubleAdd(int *in, int *partialSums, int n) {
    __shared__ int warpResults[32]; // Shared memory for warp-level results (32 warps per block)

    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    int sum = 0;

    // Perform global memory load and sum two elements per thread
    if (gid < n) {
        sum = in[gid];
        if (gid + blockDim.x < n) { // Boundary check for the second element
            sum += in[gid + blockDim.x];
        }
    }

    // Perform warp-level reduction using shuffle instructions
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Store the result of each warp in shared memory
    if ((tid & 31) == 0) {
        warpResults[tid >> 5] = sum;
    }

    __syncthreads();

    // Perform block-level reduction on warp results using a single warp
    if (tid < 32) {
        sum = (tid < blockDim.x / 32) ? warpResults[tid] : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (tid == 0) {
            partialSums[blockIdx.x] = sum;
        }
    }
}

template <unsigned int blockSize>
__global__ void warpShuffleReductionQuadAdd(int *in, int *partialSums, int n) {
    __shared__ int warpResults[32]; // Shared memory for warp-level results (32 warps per block)

    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    int sum = 0;

    // Calculate the base index for this thread
    int baseIndex = gid * 4;

    // Load 4 elements using int4 vectorized access
    int4 data = reinterpret_cast<int4*>(in)[baseIndex / 4];
    sum += data.x;
    sum += data.y;
    sum += data.z;
    sum += data.w;

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
        sum = (tid < blockDim.x / 32) ? warpResults[tid] : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (tid == 0) {
            partialSums[blockIdx.x] = sum; // Write the final block sum to global memory
        }
    }
}



template <unsigned int blockSize>
__global__ void warpShuffleReductionEightAdd(int *in, int *partialSums, int n) {
    __shared__ int warpResults[32]; // Shared memory for warp-level results (32 warps per block)

    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    int sum = 0;

    sum += in[gid] + in[gid + blockDim.x] +
     in[gid + 2 * blockDim.x] + in[gid + 3 * blockDim.x] +
      in[gid + 4 * blockDim.x] + in[gid + 5 * blockDim.x] + 
      in[gid + 6 * blockDim.x] + in[gid + 7 * blockDim.x];

    // Perform warp-level reduction using shuffle instructions
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Store the result of each warp in shared memory
    if ((tid & 31) == 0) {
        warpResults[tid >> 5] = sum;
    }

    __syncthreads();

    // Perform block-level reduction on warp results using a single warp
    if (tid < 32) {
        sum = (tid < (blockSize >> 5)) ? warpResults[tid] : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (tid == 0) {
            partialSums[blockIdx.x] = sum;
        }
    }
}

template <unsigned int blockSize>
__global__ void warpShuffleReductionEightAddLoop(int *in, int *partialSums, int n) {
    __shared__ int warpResults[32]; // Shared memory for warp-level results (32 warps per block)

    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    int sum = 0;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int index = gid + i * blockDim.x;
        if (index < n) {
            sum += in[index];
        } else {
            break; // Exit the loop early if out of bounds
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
        sum = (tid < blockDim.x / 32) ? warpResults[tid] : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (tid == 0) {
            partialSums[blockIdx.x] = sum; // Write the final block sum to global memory
        }
    }
}

template <unsigned int blockSize>
__global__ void warpShuffleReductionSixteenAdd(int *in, int *partialSums, int n) {
    __shared__ int warpResults[32]; // Shared memory for warp-level results (32 warps per block)

    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x * 16 + threadIdx.x;

    int sum = in[gid] + in[gid + blockDim.x] +
     in[gid + 2 * blockDim.x] + in[gid + 3 * blockDim.x] +
      in[gid + 4 * blockDim.x] + in[gid + 5 * blockDim.x] + 
      in[gid + 6 * blockDim.x] + in[gid + 7 * blockDim.x] +
      in[gid + 8 * blockDim.x] + in[gid + 9 * blockDim.x] +
      in[gid + 10 * blockDim.x] + in[gid + 11 * blockDim.x] +
      in[gid + 12 * blockDim.x] + in[gid + 13 * blockDim.x] +
      in[gid + 14 * blockDim.x] + in[gid + 15 * blockDim.x];

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
        sum = (tid < blockDim.x / 32) ? warpResults[tid] : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (tid == 0) {
            partialSums[blockIdx.x] = sum;
        }
    }
}

template <unsigned int blockSize>
__global__ void warpShuffleReductionThirtyTwoAdd(int *in, int *partialSums, int n) {
    __shared__ int warpResults[32]; // Shared memory for warp-level results (32 warps per block)

    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x * 32 + threadIdx.x;

    int sum = 0;

    // Perform global memory load and sum 32 elements per thread explicitly
    sum += in[gid];
    sum += in[gid + blockDim.x];
    sum += in[gid + 2 * blockDim.x];
    sum += in[gid + 3 * blockDim.x];
    sum += in[gid + 4 * blockDim.x];
    sum += in[gid + 5 * blockDim.x];
    sum += in[gid + 6 * blockDim.x];
    sum += in[gid + 7 * blockDim.x];
    sum += in[gid + 8 * blockDim.x];
    sum += in[gid + 9 * blockDim.x];
    sum += in[gid + 10 * blockDim.x];
    sum += in[gid + 11 * blockDim.x];
    sum += in[gid + 12 * blockDim.x];
    sum += in[gid + 13 * blockDim.x];
    sum += in[gid + 14 * blockDim.x];
    sum += in[gid + 15 * blockDim.x];
    sum += in[gid + 16 * blockDim.x];
    sum += in[gid + 17 * blockDim.x];
    sum += in[gid + 18 * blockDim.x];
    sum += in[gid + 19 * blockDim.x];
    sum += in[gid + 20 * blockDim.x];
    sum += in[gid + 21 * blockDim.x];
    sum += in[gid + 22 * blockDim.x];
    sum += in[gid + 23 * blockDim.x];
    sum += in[gid + 24 * blockDim.x];
    sum += in[gid + 25 * blockDim.x];
    sum += in[gid + 26 * blockDim.x];
    sum += in[gid + 27 * blockDim.x];
    sum += in[gid + 28 * blockDim.x];
    sum += in[gid + 29 * blockDim.x];
    sum += in[gid + 30 * blockDim.x];
    sum += in[gid + 31 * blockDim.x];

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
        sum = (tid < blockDim.x / 32) ? warpResults[tid] : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (tid == 0) {
            partialSums[blockIdx.x] = sum; // Write the final block sum to global memory
        }
    }
}

template <unsigned int blockSize>
__global__ void warpShuffleReductionSixteenAddLoop(int *in, int *partialSums, int n) {
    __shared__ int warpResults[32]; // Shared memory for warp-level results (32 warps per block)

    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x * 16 + threadIdx.x;

    int sum = 0;

    // Perform global memory load and sum 16 elements per thread using a loop
    for (int i = 0; i < 16; i++) {
        int index = gid + i * blockDim.x;
        if (index < n) {
            sum += in[index];
        } else {
            break; // Exit the loop early if out of bounds
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
        sum = (tid < blockDim.x / 32) ? warpResults[tid] : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (tid == 0) {
            partialSums[blockIdx.x] = sum; // Write the final block sum to global memory
        }
    }
}

template <unsigned int blockSize>
__global__ void gridStrideReductionWithShuffles(int *in, int *partialSums, int n) {
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * (blockSize * 2) + threadIdx.x;
    unsigned int gridSize = blockDim.x * 2 * gridDim.x;

    int sum = 0;

    // Perform grid-stride loop to sum elements
    while (gid < n) {
        sum += in[gid];
        if (gid + blockSize < n) { // Boundary check for the second element
            sum += in[gid + blockSize];
        }
        gid += gridSize;
    }

    // Perform warp-level reduction using shuffle instructions
    for (int offset = blockSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Write the result of the warp leader to global memory
    if (tid % 32 == 0) {
        partialSums[blockIdx.x] = sum;
    }
}

template <unsigned int blockSize>
__global__ void warpShuffleReductionVectorized(int *in, int *partialSums, int n) {
    __shared__ int warpResults[32]; // Shared memory for warp-level results (32 warps per block)

    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    int sum = 0;

    // Calculate the base index for this thread
    int baseIndex = gid * 4;

    // Load and sum 8 int4 vectors (32 integers)
    for (int i = 0; i < 8; i++) {
        int index = baseIndex + i * blockDim.x * 4; // Step by blockDim.x * 4 for each int4
        if (index < n) {
            int4 data = reinterpret_cast<int4*>(in)[index / 4];
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
        sum = (tid < blockDim.x / 32) ? warpResults[tid] : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (tid == 0) {
            partialSums[blockIdx.x] = sum; // Write the final block sum to global memory
        }
    }
}

// Kernel to sum the partial results from all blocks
__global__ void finalReduction(int *partialSums, int *result, int n) {
    int sum = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        sum += partialSums[i];
    }

    // Perform warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Store the final result
    if (threadIdx.x == 0) {
        atomicAdd(result, sum);
    }
}

__global__ void finalReductionVectorized(int *partialSums, int *result, int n) {
    int sum = 0;

    // Each thread processes 4 elements at a time using int4
    int tid = threadIdx.x;
    int numThreads = blockDim.x;
    int numVectors = n / 4; // Number of int4 vectors
    int remainder = n % 4; // Remaining elements

    for (int i = tid; i < numVectors; i += numThreads) {
        int4 data = reinterpret_cast<int4*>(partialSums)[i];
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

void displayAndVerifyResults(const std::vector<std::tuple<std::string, int, float>> &results, int cpu_result) {
    bool all_match = true;

    for (const auto &[technique, result, duration] : results) {
        std::cout << "Technique: " << technique << ", Result: " << result
                  << ", Time: " << duration << " ms" << std::endl;

        if (result != cpu_result) {
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

template <typename Kernel>
std::tuple<int, float> benchmarkKernel(Kernel kernel, int *dev_input_data, int *dev_partial_sums, int *dev_result, int n, int blockSize, int factor) {
    // Calculate the number of blocks
    int numBlocks = (n + blockSize * factor - 1) / (blockSize * factor);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timing
    cudaEventRecord(start);
    kernel<<<numBlocks, blockSize>>>(dev_input_data, dev_partial_sums, n);
    finalReductionVectorized<<<1, 32>>>(dev_partial_sums, dev_result, numBlocks);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float duration = 0.0f;
    cudaEventElapsedTime(&duration, start, stop);

    // Retrieve the result from the device
    int result;
    cudaMemcpy(&result, dev_result, sizeof(int), cudaMemcpyDeviceToHost);

    // Reset device result memory
    cudaMemset(dev_result, 0, sizeof(int));

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return {result, duration};
}

int main() {
    int n = 1 << 25; // 4M elements
    size_t bytes = n * sizeof(int);

    // Host/CPU arrays
    int *host_input_data = new int[n];

    // Device/GPU arrays
    int *dev_input_data, *dev_partial_sums, *dev_result;

    // Initialize data
    srand(42); // Fixed seed
    for (int i = 0; i < n; i++) {
        host_input_data[i] = rand() % 100;
    }

    // Allocate memory on GPU
    cudaMalloc(&dev_input_data, bytes);
    cudaMalloc(&dev_partial_sums, ((n + 1023) / 1024) * sizeof(int));
    cudaMalloc(&dev_result, sizeof(int));
    cudaMemset(dev_result, 0, sizeof(int));

    // Copy data to GPU
    cudaMemcpy(dev_input_data, host_input_data, bytes, cudaMemcpyHostToDevice);

    int blockSize = 1024; // Threads per block

    // Map-like structure to store results and durations
    std::vector<std::tuple<std::string, int, float>> results;

    // Perform CPU reduction
    auto cpu_start = std::chrono::high_resolution_clock::now();
    int cpu_result = std::accumulate(host_input_data, host_input_data + n, 0);
    auto cpu_stop = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_stop - cpu_start).count() / 1000.0;
    results.emplace_back("CPU", cpu_result, cpu_duration);

    // Benchmark different kernels
    {
        auto [result, duration] = benchmarkKernel(warpShuffleReduction<1024>, dev_input_data, dev_partial_sums, dev_result, n, blockSize, 1);
        results.emplace_back("Restored", result, duration);
    }
    {
        auto [result, duration] = benchmarkKernel(warpShuffleReductionDoubleAdd<1024>, dev_input_data, dev_partial_sums, dev_result, n, blockSize, 2);
        results.emplace_back("Double-Add", result, duration);
    }
    {
        auto [result, duration] = benchmarkKernel(warpShuffleReductionQuadAdd<1024>, dev_input_data, dev_partial_sums, dev_result, n, blockSize, 4);
        results.emplace_back("Quad-Add", result, duration);
    }
    {
        auto [result, duration] = benchmarkKernel(warpShuffleReductionEightAdd<1024>, dev_input_data, dev_partial_sums, dev_result, n, blockSize, 8);
        results.emplace_back("Eight-Add", result, duration);
    }
    {
        auto [result, duration] = benchmarkKernel(warpShuffleReductionSixteenAdd<1024>, dev_input_data, dev_partial_sums, dev_result, n, blockSize, 16);
        results.emplace_back("Sixteen-Add", result, duration);
    }
    {
        auto [result, duration] = benchmarkKernel(warpShuffleReductionThirtyTwoAdd<1024>, dev_input_data, dev_partial_sums, dev_result, n, blockSize, 32);
        results.emplace_back("Thirty-Two-Add", result, duration);
    }
    {
        auto [result, duration] = benchmarkKernel(warpShuffleReductionVectorized<1024>, dev_input_data, dev_partial_sums, dev_result, n, blockSize, 32);
        results.emplace_back("Vectorized", result, duration);
    }

    // CUB reduction
    int *dev_cub_result;
    cudaMalloc(&dev_cub_result, sizeof(int));
    cudaMemset(dev_cub_result, 0, sizeof(int)); // Initialize to 0

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
    int cub_result;
    cudaMemcpy(&cub_result, dev_cub_result, sizeof(int), cudaMemcpyDeviceToHost);
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