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

void fill_matrix(int *mat, int N) {
    for (int i = 0; i < N * N; ++i) {
        mat[i] = rand() % 10;
    }
}

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : 512;
    size_t size = N * N * sizeof(int);

    // Allocate pinned host memory for faster transfer
    int *h_A, *h_B, *h_C;
    cudaMallocHost(&h_A, size);
    cudaMallocHost(&h_B, size);
    cudaMallocHost(&h_C, size);

    fill_matrix(h_A, N);
    fill_matrix(h_B, N);

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x,
                (N + threads.y - 1) / threads.y);

    auto start_gpu = std::chrono::high_resolution_clock::now();
    naiveMatMul<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("Naive Matrix Multiplication (%d x %d)\n", N, N);
    printf("GPU Time: %f seconds\n",
           std::chrono::duration<double>(end_gpu - start_gpu).count());
    printf("Sample result: C[0][0] = %d\n", h_C[0]);

    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
