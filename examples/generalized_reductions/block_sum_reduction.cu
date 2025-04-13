#include <stdio.h>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <numeric>  // For std::accumulate
#include <limits>
#include  <algorithm>

#define BLOCK_SIZE 256

using namespace std;

// Fill array with random integers
void fill_array(int N, int* x) {
    for (int i = 0; i < N; i++) {
        x[i] = rand() % 100;
    }
}

// Operator structs
struct AddOp {
    __device__ __host__ int operator()(int a, int b) const { return a + b; }
};

struct MaxOp {
    __device__ __host__ int operator()(int a, int b) const { return a > b ? a : b; }
};

struct MinOp {
    __device__ __host__ int operator()(int a, int b) const { return a < b ? a : b; }
};

// Generalized block reduction kernel
template <typename T, typename Op>
__global__ void blockReduction(T *in, T *out, int n, Op op, T identity) {
    extern __shared__ T subArray[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    T val = identity;
    if (gid < n) val = op(val, in[gid]);
    if (gid + blockDim.x < n) val = op(val, in[gid + blockDim.x]);
    subArray[tid] = val;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            subArray[tid] = op(subArray[tid], subArray[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = subArray[0];
    }
}

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : (1 << 26);  // 64M ints
    string op_name = (argc > 2) ? argv[2] : "sum";

    cout << "Running generalized block reduction (" << op_name << ") on " << N << " elements\n";

    // --- Host allocation and initialization
    int* h_input = (int*)malloc(N * sizeof(int));
    fill_array(N, h_input);

    // --- Device allocations
    int *d_input, *d_partial;
    cudaMalloc(&d_input, N * sizeof(int));

    int THREAD_COUNT = BLOCK_SIZE;
    int BLOCK_COUNT = (N + THREAD_COUNT * 2 - 1) / (THREAD_COUNT * 2);
    cudaMalloc(&d_partial, BLOCK_COUNT * sizeof(int));

    // --- Copy input to device
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    // --- GPU Reduction
    auto start_gpu = chrono::high_resolution_clock::now();

    if (op_name == "sum") {
        blockReduction<int, AddOp><<<BLOCK_COUNT, THREAD_COUNT, THREAD_COUNT * sizeof(int)>>>(d_input, d_partial, N, AddOp(), 0);
    } else if (op_name == "max") {
        blockReduction<int, MaxOp><<<BLOCK_COUNT, THREAD_COUNT, THREAD_COUNT * sizeof(int)>>>(d_input, d_partial, N, MaxOp(), INT_MIN);
    } else if (op_name == "min") {
        blockReduction<int, MinOp><<<BLOCK_COUNT, THREAD_COUNT, THREAD_COUNT * sizeof(int)>>>(d_input, d_partial, N, MinOp(), INT_MAX);
    } else {
        cerr << "Unknown operation: " << op_name << endl;
        return 1;
    }

    cudaDeviceSynchronize();
    auto end_gpu = chrono::high_resolution_clock::now();

    // --- Copy back partial sums and finalize on CPU
    int* h_partial = (int*)malloc(BLOCK_COUNT * sizeof(int));
    cudaMemcpy(h_partial, d_partial, BLOCK_COUNT * sizeof(int), cudaMemcpyDeviceToHost);

    int gpu_result = (op_name == "sum") ? 0 : 
                     (op_name == "max") ? INT_MIN : INT_MAX;

    if (op_name == "sum") {
        for (int i = 0; i < BLOCK_COUNT; ++i) gpu_result += h_partial[i];
    } else if (op_name == "max") {
        for (int i = 0; i < BLOCK_COUNT; ++i) gpu_result = max(gpu_result, h_partial[i]);
    } else if (op_name == "min") {
        for (int i = 0; i < BLOCK_COUNT; ++i) gpu_result = min(gpu_result, h_partial[i]);
    }

    chrono::duration<double> gpu_time = end_gpu - start_gpu;

    // --- Optional CPU comparison
    int cpu_result = (op_name == "sum") ? accumulate(h_input, h_input + N, 0) :
                     (op_name == "max") ? *max(h_input, h_input + N) :
                     (op_name == "min") ? *min(h_input, h_input + N) : -1;

    // --- Output
    cout << "GPU reduction result       : " << gpu_result << endl;
    cout << "CPU verification result    : " << cpu_result << endl;
    cout << "GPU reduction time         : " << gpu_time.count() << " seconds\n";

    // --- Cleanup
    free(h_input);
    free(h_partial);
    cudaFree(d_input);
    cudaFree(d_partial);

    return 0;
}
