// naive_matrix_multiplication.cu
#include <stdio.h>
#include <cuda.h>
#include <chrono>

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

int main() {
    const int N = 1024; // Matrix size N x N
    size_t size = N * N * sizeof(int);

    int *h_A = (int *)malloc(size);
    int *h_B = (int *)malloc(size);
    int *h_C_cpu = (int *)malloc(size);
    int *h_C_gpu = (int *)malloc(size);

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1;
        h_B[i] = 1;
    }

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);

    auto start_gpu = std::chrono::high_resolution_clock::now();
    naiveMatMul<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);

    auto start_cpu = std::chrono::high_resolution_clock::now();
    cpuMatMul(h_A, h_B, h_C_cpu, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();

    printf("Naive GPU Time: %f seconds\n", std::chrono::duration<double>(end_gpu - start_gpu).count());
    printf("CPU Time:       %f seconds\n", std::chrono::duration<double>(end_cpu - start_cpu).count());
    printf("Result check:   C[0][0] = %d (GPU), %d (CPU)\n", h_C_gpu[0], h_C_cpu[0]);

    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
