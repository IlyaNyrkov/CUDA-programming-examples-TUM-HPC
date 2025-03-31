// naive_matrix_multiplication.cu
#include <stdio.h>
#include <cuda.h>
#include <chrono>

__global__ void warpTiledMatMul(int* A, int* B, int* C, int N) {
    __shared__ int sharedA[32][32];
    __shared__ int sharedB[32][32];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * 32 + ty;
    int col = blockIdx.x * 32 + tx;

    int sum = 0;
    for (int tile = 0; tile < (N + 15) / 32; tile++) {
        if (row < N && tile * 32 + tx < N)
            sharedA[ty][tx] = A[row * N + tile * 32 + tx];
        else
            sharedA[ty][tx] = 0;

        if (col < N && tile * 32 + ty < N)
            sharedB[ty][tx] = B[(tile * 32 + ty) * N + col];
        else
            sharedB[ty][tx] = 0;

        __syncthreads();

        for (int k = 0; k < 32; k++) {
            sum += sharedA[ty][k] * sharedB[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

void cpuMatMul(int *A, int *B, int *C, int N) {
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            int sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

bool compareMatricies(int *left, int* right, int N, int M) {
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            if (left[row * N + col] != right[row * N + col])  {
                return false;
            }
        }
    }

    return true;
}

int main() {
    const int N = 1024; // Matrix size N x N
    size_t size = N * N * sizeof(int);

    int *matrixACpu = (int *)malloc(size);
    int *matrixBCpu = (int *)malloc(size);
    int *resultMatrixCpu_cpu = (int *)malloc(size);
    int *resultMatrixCpu_gpu = (int *)malloc(size);

    int *matrixAGpu, *matrixBGpu, *matrixCGpu;
    cudaMalloc(&matrixAGpu, size);
    cudaMalloc(&matrixBGpu, size);
    cudaMalloc(&matrixCGpu, size);

    for (int i = 0; i < N * N; i++) {
        matrixACpu[i] = 1;
        matrixBCpu[i] = 1;
    }

    cudaMemcpy(matrixAGpu, matrixACpu, size, cudaMemcpyHostToDevice);
    cudaMemcpy(matrixBGpu, matrixBCpu, size, cudaMemcpyHostToDevice);

    dim3 threads(32, 32);
    dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);

    auto start_gpu = std::chrono::high_resolution_clock::now();
    warpTiledMatMul<<<blocks, threads>>>(matrixAGpu, matrixBGpu, matrixCGpu, N);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();

    cudaMemcpy(resultMatrixCpu_gpu, matrixCGpu, size, cudaMemcpyDeviceToHost);

    auto start_cpu = std::chrono::high_resolution_clock::now();
    cpuMatMul(matrixACpu, matrixBCpu, resultMatrixCpu_cpu, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();

    printf("Matrix %d x %d\n", N, N);
    printf("Warp optimized Method GPU Time: %f seconds\n", std::chrono::duration<double>(end_gpu - start_gpu).count());
    printf("CPU Time:       %f seconds\n", std::chrono::duration<double>(end_cpu - start_cpu).count());
    printf("Result check:   C[0][0] = %d (GPU), %d (CPU)\n", resultMatrixCpu_gpu[0], resultMatrixCpu_cpu[0]);
    printf("CPU and gpu matricies same: %d\n", compareMatricies(resultMatrixCpu_cpu, resultMatrixCpu_gpu, N, N));

    free(matrixACpu);
    free(matrixBCpu);
    free(resultMatrixCpu_cpu);
    free(resultMatrixCpu_gpu);
    cudaFree(matrixAGpu);
    cudaFree(matrixBGpu);
    cudaFree(matrixCGpu);
    return 0;
}
