#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <chrono>

#define TILE_WIDTH 16

void matMulCPU(int* A, int* B, int* C, int N) {
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

__global__ void tiledMatMul(int* A, int* B, int* C, int N) {
    __shared__ int tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ int tileB[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    int sum = 0;
    for (int tile = 0; tile < (N + TILE_WIDTH - 1) / TILE_WIDTH; tile++) {
        if (row < N && tile * TILE_WIDTH + tx < N)
            tileA[ty][tx] = A[row * N + tile * TILE_WIDTH + tx];
        else
            tileA[ty][tx] = 0;

        if (col < N && tile * TILE_WIDTH + ty < N)
            tileB[ty][tx] = B[(tile * TILE_WIDTH + ty) * N + col];
        else
            tileB[ty][tx] = 0;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += tileA[ty][k] * tileB[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

int main() {
    const int N = 1024;
    size_t bytes = N * N * sizeof(int);

    int* h_A = (int*)malloc(bytes);
    int* h_B = (int*)malloc(bytes);
    int* h_C_CPU = (int*)malloc(bytes);
    int* h_C_GPU = (int*)malloc(bytes);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1;
        h_B[i] = 1;
    }

    auto start_cpu = std::chrono::high_resolution_clock::now();
    matMulCPU(h_A, h_B, h_C_CPU, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_time = end_cpu - start_cpu;

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    // ---------- GPU Run ----------
    auto start_gpu = std::chrono::high_resolution_clock::now();
    tiledMatMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_time = end_gpu - start_gpu;

    cudaMemcpy(h_C_GPU, d_C, bytes, cudaMemcpyDeviceToHost);

    // ---------- Compare ----------
    bool correct = true;
    for (int i = 0; i < N * N; i++) {
        if (h_C_CPU[i] != h_C_GPU[i]) {
            correct = false;
            break;
        }
    }

    printf("Tiled Matrix Multiplication (N = %d)\n", N);
    printf("CPU Time: %.6f seconds\n", cpu_time.count());
    printf("GPU Time: %.6f seconds\n", gpu_time.count());
    printf("Results match: %s\n", correct ? "YES" : "NO");

    // ---------- Cleanup ----------
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_CPU); free(h_C_GPU);
    return 0;
}
