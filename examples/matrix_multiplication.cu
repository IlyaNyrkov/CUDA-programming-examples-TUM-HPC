#include <stdio.h>
#include <cuda.h>
#include <chrono>

#define CHECK_CUDA(call)                                                    \
    {                                                                       \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            printf("CUDA error %d at %s:%d: %s\n", err, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    }

// Kernel 1: Naive matrix multiplication (worst method)
__global__ void naiveMatMul(int *A, int *B, int *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int sum = 0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Kernel 2: Shared memory optimization
__global__ void sharedMatMul(int *A, int *B, int *C, int N) {
    __shared__ int sharedA[16][16];
    __shared__ int sharedB[16][16];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;

    int sum = 0;
    for (int tile = 0; tile < (N + 15) / 16; tile++) {
        if (row < N && tile * 16 + tx < N) {
            sharedA[ty][tx] = A[row * N + tile * 16 + tx];
        } else {
            sharedA[ty][tx] = 0;
        }

        if (col < N && tile * 16 + ty < N) {
            sharedB[ty][tx] = B[(tile * 16 + ty) * N + col];
        } else {
            sharedB[ty][tx] = 0;
        }

        __syncthreads();

        for (int k = 0; k < 16; k++) {
            sum += sharedA[ty][k] * sharedB[k][tx];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Kernel 3: Warp-optimized matrix multiplication (best method)
__global__ void warpOptimizedMatMul(int *A, int *B, int *C, int N) {
    __shared__ int sharedA[32][32];
    __shared__ int sharedB[32][32];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;

    int sum = 0;
    for (int tile = 0; tile < (N + 31) / 32; tile++) {
        if (row < N && tile * 32 + tx < N) {
            sharedA[ty][tx] = A[row * N + tile * 32 + tx];
        } else {
            sharedA[ty][tx] = 0;
        }

        if (col < N && tile * 32 + ty < N) {
            sharedB[ty][tx] = B[(tile * 32 + ty) * N + col];
        } else {
            sharedB[ty][tx] = 0;
        }

        __syncthreads();

        for (int k = 0; k < 32; k++) {
            sum += sharedA[ty][k] * sharedB[k][tx];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

void runAndTimeMatMul(void (*kernel)(int *, int *, int *, int), const char *methodName, int *d_A, int *d_B, int *d_C, int N) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);

    // Reset output
    CHECK_CUDA(cudaMemset(d_C, 0, N * N * sizeof(int)));

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Launch kernel
    kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Copy result back to host
    int *h_C = (int *)malloc(N * N * sizeof(int));
    CHECK_CUDA(cudaMemcpy(h_C, d_C, N * N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print some results for verification
    printf("%s: C[0][0] = %d, Time = %f seconds\n", methodName, h_C[0], elapsed.count());

    free(h_C);
}

int main() {
    const int N = 5000; // Matrix size N x N
    int *h_A, *h_B;
    int *d_A, *d_B, *d_C;

    // Allocate host memory
    h_A = (int *)malloc(N * N * sizeof(int));
    h_B = (int *)malloc(N * N * sizeof(int));

    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1; // Initialize all elements to 1
        h_B[i] = 1; // Initialize all elements to 1
    }

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_A, N * N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_B, N * N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_C, N * N * sizeof(int)));

    // Copy input data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, N * N * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, N * N * sizeof(int), cudaMemcpyHostToDevice));

    // Run and time kernels
    runAndTimeMatMul(naiveMatMul, "Naive Matrix Multiplication", d_A, d_B, d_C, N);
    runAndTimeMatMul(sharedMatMul, "Shared Memory Matrix Multiplication", d_A, d_B, d_C, N);
    runAndTimeMatMul(warpOptimizedMatMul, "Warp-Optimized Matrix Multiplication", d_A, d_B, d_C, N);

    // Cleanup
    free(h_A);
    free(h_B);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}