#include <iostream>
#include <cuda_runtime.h>
#include <cub/cub.cuh> // Include CUB library
#include <chrono>
#include <numeric>
using namespace std;

// Global constant for the number of runs
const int NUM_RUNS = 50;

template <unsigned int blockSize>
__global__ void __launch_bounds__(1024) warpShuffleReduction(int *__restrict__ in, int *__restrict__ partialSums, int n) {
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
__global__ void __launch_bounds__(1024) warpShuffleReductionDoubleAdd(int *__restrict__ in, int *__restrict__ partialSums, int n) {
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
__global__ void __launch_bounds__(1024) warpShuffleReductionQuadAdd(int *__restrict__ in, int *__restrict__ partialSums, int n) {
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
__global__ void __launch_bounds__(1024) warpShuffleReductionEightAdd(int *__restrict__ in, int *__restrict__ partialSums, int n) {
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
__global__ void __launch_bounds__(1024) warpShuffleReductionEightAddLoop(int *__restrict__ in, int *__restrict__ partialSums, int n) {
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
__global__ void __launch_bounds__(1024) warpShuffleReductionSixteenAdd(int *__restrict__ in, int *__restrict__ partialSums, int n) {
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
__global__ void __launch_bounds__(1024) warpShuffleReductionThirtyTwoAdd(int *__restrict__ in, int *__restrict__ partialSums, int n) {
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
__global__ void __launch_bounds__(1024) warpShuffleReductionSixteenAddLoop(int *__restrict__ in, int *__restrict__ partialSums, int n) {
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
__global__ void __launch_bounds__(1024) gridStrideReductionWithShuffles(int *__restrict__ in, int *__restrict__ partialSums, int n) {
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
__global__ void __launch_bounds__(1024) warpShuffleReductionVectorized(int *__restrict__ in, int *__restrict__ partialSums, int n) {
    __shared__ int warpResults[32]; // Shared memory for warp-level results (32 warps per block)

    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockSize * 8 + threadIdx.x;

    int sum = 0;

    // Calculate the base index for this thread
    int baseIndex = gid * 4;

    // Load and sum 8 int4 vectors (32 integers)
    for (int i = 0; i < 8; i++) {
        int index = baseIndex + i * blockSize * 4; // Step by blockDim.x * 4 for each int4
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
        sum = (tid < blockSize / 32) ? warpResults[tid] : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (tid == 0) {
            partialSums[blockIdx.x] = sum; // Write the final block sum to global memory
        }
    }
}

template <unsigned int blockSize>
__global__ void __launch_bounds__(1024) warpShuffleReductionSequentialReads(int *__restrict__ in, int *__restrict__ partialSums, int n) {
    __shared__ int warpResults[32]; // Shared memory for warp-level results (32 warps per block)

    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockSize * 8 + threadIdx.x;

    int sum = 0;

    // Calculate the base index for this thread
    int baseIndex = gid * 4; // Each thread processes 32 elements sequentially

    for (int i = 0; i < 8; i++) {
        int index = baseIndex + i * blockSize * 4; // Step by blockDim.x * 4 for each 4 integers

        if (index < n) {
            sum += in[index];
            sum += in[index + 1];
            sum += in[index + 2];
            sum += in[index + 3];
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
        sum = (tid < blockSize / 32) ? warpResults[tid] : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (tid == 0) {
            partialSums[blockIdx.x] = sum; // Write the final block sum to global memory
        }
    }
}

// Kernel to sum the partial results from all blocks
__global__ void finalReduction(int *__restrict__ partialSums, int *__restrict__ result, int n) {
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

struct int8 {
    int x, y, z, w, a, b, c, d;
};

template <unsigned int blockSize>
__global__ void __launch_bounds__(1024) warpShuffleReductionCustomStruct(int *__restrict__ in, int *__restrict__ partialSums, int n) {
    __shared__ int warpResults[32]; // Shared memory for warp-level results (32 warps per block)

    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockSize * 4 + threadIdx.x;

    int sum = 0;

    // Calculate the base index for this thread
    int baseIndex = gid * 8; // Each thread processes 8 integers sequentially

    // Load and sum 8 integers using the custom int8 structure
    for (int i = 0; i < 4; i++) {
        int index = baseIndex + i * blockSize * 8; // Step by blockDim.x * 8 for each int8

        if (index < n) {
            int8 data = reinterpret_cast<int8*>(in)[index / 8];
            sum += data.x + data.y + data.z + data.w + data.a + data.b + data.c + data.d;
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
        sum = (tid < blockSize / 32) ? warpResults[tid] : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (tid == 0) {
            partialSums[blockIdx.x] = sum; // Write the final block sum to global memory
        }
    }
}

template <unsigned int blockSize>
__global__ void __launch_bounds__(1024) warpShuffleReduction64Elements(int *__restrict__ in, int *__restrict__ partialSums, int n) {
    __shared__ int warpResults[32]; // Shared memory for warp-level results (32 warps per block)

    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockSize * 16 + threadIdx.x; // Each thread processes 64 elements (16 * int4)

    int sum = 0;

    // Calculate the base index for this thread
    int baseIndex = gid * 4; // Each thread processes 4 integers per iteration

    // Load and sum 16 int4 vectors (64 integers)
    for (int i = 0; i < 16; i++) {
        int index = baseIndex + i * blockSize * 4; // Step by blockDim.x * 4 for each int4
        if (index < n) {
            int4 data = reinterpret_cast<int4*>(in)[index / 4];
            sum += data.x + data.y + data.z + data.w;
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
        sum = (tid < blockSize / 32) ? warpResults[tid] : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (tid == 0) {
            partialSums[blockIdx.x] = sum; // Write the final block sum to global memory
        }
    }
}

__global__ void finalReductionVectorized(int *__restrict__ partialSums, int *__restrict__ result, int n) {
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

template <unsigned int blockSize, unsigned int elementsPerThread>
__global__ void __launch_bounds__(1024) warpShuffleReductionGeneralized(int *__restrict__ in, int *__restrict__ partialSums, int n) {
    __shared__ int warpResults[32]; // Shared memory for warp-level results (32 warps per block)

    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockSize * elementsPerThread + threadIdx.x;

    int sum = 0;

    // Calculate the base index for this thread
    int baseIndex = gid;

    // Load and sum `elementsPerThread` elements sequentially
    #pragma unroll
    for (int i = 0; i < elementsPerThread; i++) {
        int index = baseIndex + i * blockSize; // Step by blockDim.x for each element
        if (index < n) {
            sum += in[index];
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
        sum = (tid < blockSize / 32) ? warpResults[tid] : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (tid == 0) {
            partialSums[blockIdx.x] = sum; // Write the final block sum to global memory
        }
    }
}

template <unsigned int blockSize, unsigned int elementsPerThread>
__global__ void __launch_bounds__(1024) 
warpShuffleReductionVectorizedGeneralized(int *__restrict__ in, int *__restrict__ partialSums, int n) 
{
    __shared__ int warpResults[32];
    const unsigned int tid = threadIdx.x;
    const unsigned int gid = blockIdx.x * blockSize * elementsPerThread/4 + tid;
    
    // Use vector types (process 4 elements per load)
    constexpr int vectorSize = 4;
    constexpr int vectorsPerThread = (elementsPerThread + vectorSize - 1) / vectorSize;
    
    int sum = 0;

    #pragma unroll
    for (int i = 0; i < vectorsPerThread; i++) 
    {
        int index = (gid + i * blockSize) * vectorSize;
        
        if (index + vectorSize - 1 < n) {
            // Full vector load
            int4 vec = reinterpret_cast<int4*>(in)[index / vectorSize];
            sum += vec.x + vec.y + vec.z + vec.w;
        }
        else if (index < n) {
            // Partial vector load (boundary condition)
            int *ptr = in + index;
            #pragma unroll
            for (int j = 0; j < vectorSize && (index + j) < n; j++) {
                sum += ptr[j];
            }
        }
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Store warp results
    if (tid % 32 == 0) {
        warpResults[tid / 32] = sum;
    }

    __syncthreads();

    // Block-level reduction
    if (tid < 32) {
        sum = (tid < blockSize / 32) ? warpResults[tid] : 0;
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        if (tid == 0) {
            partialSums[blockIdx.x] = sum;
        }
    }
}

template <unsigned int blockSize, unsigned int elementsPerThread>
__global__ void __launch_bounds__(1024) warpShufflePrefetchReduction(int *__restrict__ in, int *__restrict__ partialSums, int n) 
{
    __shared__ int warpResults[32]; // 1 per warp

    const unsigned int tid = threadIdx.x;
    const unsigned int gid = blockIdx.x * blockSize * elementsPerThread + tid;

    // Four independent accumulators
    int sum0 = 0, sum1 = 0;
    // Software pipelining with prefetch
    #pragma unroll(2)
    for (int i = 0; i < elementsPerThread; i += 2) 
    {
        // Prefetch next 4 elements (coalesced reads)
        int index0 = gid + (i+0)*blockSize;
        int index1 = gid + (i+1)*blockSize;
        
        int val0 = (index0 < n) ? in[index0] : 0;
        int val1 = (index1 < n) ? in[index1] : 0;

        // Parallel accumulation
        sum0 += val0;
        sum1 += val1;
   
    }

    // Combine accumulators
    int sum = sum0 + sum1;

    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Store warp results
    if (tid % 32 == 0) {
        warpResults[tid / 32] = sum;
    }

    __syncthreads();

    // Block-level reduction (single warp)
    if (tid < 32) {
        sum = (tid < blockSize / 32) ? warpResults[tid] : 0;
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (tid == 0) {
            partialSums[blockIdx.x] = sum;
        }
    }
}

void displayAndVerifyResults(const std::vector<std::tuple<std::string, int, float>> &results, int cpu_result) {
    bool all_match = true;

    // Find the CUB result and duration
    auto cub_entry = std::find_if(results.begin(), results.end(), [](const auto &entry) {
        return std::get<0>(entry) == "CUB";
    });

    if (cub_entry == results.end()) {
        std::cerr << "CUB result not found in results!" << std::endl;
        return;
    }

    int cub_result = std::get<1>(*cub_entry);
    float cub_duration = std::get<2>(*cub_entry);

    std::cout << "CUB Result: " << cub_result << ", Time: " << cub_duration << " ms\n";

    for (const auto &[technique, result, duration] : results) {
        std::cout << "Technique: " << technique << ", Result: " << result
                  << ", Time: " << duration << " ms";

        if (result != cpu_result) {
            all_match = false;
            std::cout << " \033[31m(Mismatch: Expected " << cpu_result << ")\033[0m";
        }

        // Compare with CUB
        if (technique != "CUB") {
            float speedup = cub_duration / duration;
            if (speedup > 1.0f) {
                std::cout << " \033[32m(Faster than CUB: " << speedup << "x)\033[0m";
            } else {
                std::cout << " \033[31m(Slower than CUB: " << 1.0f / speedup << "x)\033[0m";
            }
        }

        std::cout << std::endl;
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

    // Warmup phase
    kernel<<<numBlocks, blockSize>>>(dev_input_data, dev_partial_sums, n);
    finalReductionVectorized<<<1, 32>>>(dev_partial_sums, dev_result, numBlocks);
    cudaDeviceSynchronize();

    // Run the kernel multiple times for reliable timing
    float totalDuration = 0.0f;
    int result = 0;

    for (int i = 0; i < NUM_RUNS; i++) {
        // Reset the result memory
        cudaMemset(dev_result, 0, sizeof(int));

        // Start timing
        cudaEventRecord(start);

        // Launch the reduction kernel
        kernel<<<numBlocks, blockSize>>>(dev_input_data, dev_partial_sums, n);

        // Launch the final reduction kernel
        finalReductionVectorized<<<1, 32>>>(dev_partial_sums, dev_result, numBlocks);

        // Stop timing immediately after the final kernel launch
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // Calculate elapsed time
        float duration = 0.0f;
        cudaEventElapsedTime(&duration, start, stop);
        totalDuration += duration;
    }

    // Retrieve the result from the device
    cudaMemcpy(&result, dev_result, sizeof(int), cudaMemcpyDeviceToHost);

    // Calculate the average duration
    float averageDuration = totalDuration / NUM_RUNS;

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return {result, averageDuration};
}

int main() {
    cout << "Each kernel will be run " << NUM_RUNS << " times for reliable timing." << endl;

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
    auto cpu_start = std::chrono::steady_clock::now(); // Use steady_clock for monotonic timing
    int cpu_result = std::accumulate(host_input_data, host_input_data + n, 0);
    auto cpu_stop = std::chrono::steady_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_stop - cpu_start).count() / 1000.0;
    results.emplace_back("CPU", cpu_result, cpu_duration);

    // Benchmark different kernels
    {
        auto [result, duration] = benchmarkKernel(warpShuffleReduction<1024>, dev_input_data, dev_partial_sums, dev_result, n, blockSize, 1);
        results.emplace_back("Basic 1 element", result, duration);
    }
    {
        auto [result, duration] = benchmarkKernel(warpShuffleReductionVectorized<1024>, dev_input_data, dev_partial_sums, dev_result, n, blockSize, 32);
        results.emplace_back("32 add Vectorized", result, duration);
    }
    {
        auto [result, duration] = benchmarkKernel(warpShuffleReductionSequentialReads<1024>, dev_input_data, dev_partial_sums, dev_result, n, blockSize, 32);
        results.emplace_back("Sequential Reads", result, duration);
    }
    {
        auto [result, duration] = benchmarkKernel(warpShuffleReductionCustomStruct<1024>, dev_input_data, dev_partial_sums, dev_result, n, blockSize, 32);
        results.emplace_back("Custom Struct", result, duration);
    }
    {
        auto [result, duration] = benchmarkKernel(warpShuffleReduction64Elements<1024>, dev_input_data, dev_partial_sums, dev_result, n, blockSize, 64);
        results.emplace_back("64 Elements", result, duration);
    }
    {
        auto [result, duration] = benchmarkKernel(warpShuffleReductionGeneralized<1024, 2>, dev_input_data, dev_partial_sums, dev_result, n, blockSize, 2);
        results.emplace_back("2 Elements per Thread (Generalized kernel)", result, duration);
    }
    {
        auto [result, duration] = benchmarkKernel(warpShuffleReductionGeneralized<1024, 4>, dev_input_data, dev_partial_sums, dev_result, n, blockSize, 4);
        results.emplace_back("4 Elements per Thread (Generalized kernel)", result, duration);
    }
    {
        auto [result, duration] = benchmarkKernel(warpShuffleReductionGeneralized<1024, 8>, dev_input_data, dev_partial_sums, dev_result, n, blockSize, 8);
        results.emplace_back("8 Elements per Thread (Generalized kernel)", result, duration);
    }
    {
        auto [result, duration] = benchmarkKernel(warpShuffleReductionGeneralized<1024, 16>, dev_input_data, dev_partial_sums, dev_result, n, blockSize, 16);
        results.emplace_back("16 Elements per Thread (Generalized kernel)", result, duration);
    }
    {
        auto [result, duration] = benchmarkKernel(warpShuffleReductionGeneralized<1024, 32>, dev_input_data, dev_partial_sums, dev_result, n, blockSize, 32);
        results.emplace_back("32 Elements per Thread (Generalized kernel)", result, duration);
    }
    {
        auto [result, duration] = benchmarkKernel(warpShuffleReductionGeneralized<1024, 64>, dev_input_data, dev_partial_sums, dev_result, n, blockSize, 64);
        results.emplace_back("64 Elements per Thread (Generalized kernel)", result, duration);
    }
    {
        auto [result, duration] = benchmarkKernel(warpShuffleReductionGeneralized<1024, 128>, dev_input_data, dev_partial_sums, dev_result, n, blockSize, 128);
        results.emplace_back("128 Elements per Thread (Generalized kernel)", result, duration);
    }
    {
        auto [result, duration] = benchmarkKernel(warpShuffleReductionVectorizedGeneralized<1024, 16>, dev_input_data, dev_partial_sums, dev_result, n, blockSize, 16);
        results.emplace_back("Vectorized Generalized kernel (16 elements)", result, duration);
    }
    {
        auto [result, duration] = benchmarkKernel(warpShuffleReductionVectorizedGeneralized<1024, 32>, dev_input_data, dev_partial_sums, dev_result, n, blockSize, 32);
        results.emplace_back("Vectorized Generalized kernel (32 elements)", result, duration);
    }
    {
        auto [result, duration] = benchmarkKernel(warpShuffleReductionVectorizedGeneralized<1024, 64>, dev_input_data, dev_partial_sums, dev_result, n, blockSize, 64);
        results.emplace_back("Vectorized Generalized kernel (64 elements)", result, duration);
    }
    { 
        auto [result, duration] = benchmarkKernel(warpShuffleReductionVectorizedGeneralized<1024, 128>, dev_input_data, dev_partial_sums, dev_result, n, blockSize, 128);
        results.emplace_back("Vectorized Generalized kernel (128 elements)", result, duration); 
    }    
    {
        auto [result, duration] = benchmarkKernel(warpShufflePrefetchReduction<1024, 32>, dev_input_data, dev_partial_sums, dev_result, n, blockSize, 32);
        results.emplace_back("Prefetch Reduction (32 elems)", result, duration);
    }
    {
        auto [result, duration] = benchmarkKernel(warpShufflePrefetchReduction <1024, 64>, dev_input_data, dev_partial_sums, dev_result, n, blockSize, 64);
        results.emplace_back("Prefetch Reduction (64 elems)", result, duration);
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

    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, dev_input_data, dev_cub_result, n);
    cudaDeviceSynchronize();

    // Run the CUB reduction multiple times for reliable timing
    float totalDuration = 0.0f;

    for (int i = 0; i < NUM_RUNS; i++) {
        // Reset the result memory
        cudaMemset(dev_cub_result, 0, sizeof(int));

        // Start timing
        cudaEventRecord(start);

        // Perform CUB reduction
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, dev_input_data, dev_cub_result, n);

        // Stop timing immediately after the CUB kernel launch
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // Calculate elapsed time
        float duration = 0.0f;
        cudaEventElapsedTime(&duration, start, stop);
        totalDuration += duration;
    }

    float cub_duration = totalDuration / NUM_RUNS;
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