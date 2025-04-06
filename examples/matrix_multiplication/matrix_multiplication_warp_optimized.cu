#include <stdio.h>
#include <cuda.h>
#include <chrono>

#define TILE_SIZE 32  // Optimal for warp-level operations on modern GPUs

__global__ void warpTiledMatMul(int* A, int* B, int* C, int N) {
    __shared__ int sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ int sharedB[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    int sum = 0;
    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        if (row < N && tile * TILE_SIZE + tx < N)
            sharedA[ty][tx] = A[row * N + tile * TILE_SIZE + tx];
        else
            sharedA[ty][tx] = 0;

        if (col < N && tile * TILE_SIZE + ty < N)
            sharedB[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        else
            sharedB[ty][tx] = 0;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += sharedA[ty][k] * sharedB[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

void fill_matrix(int* mat, int N) {
    for (int i = 0; i < N * N; i++) {
        mat[i] = rand() % 10;
    }
}

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : 512;  // Default to 512x512
    size_t size = N * N * sizeof(int);

    printf("Matrix %d x %d\n", N, N);

    // Use pinned memory for better transfer speed
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

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    auto start_gpu = std::chrono::high_resolution_clock::now();
    warpTiledMatMul<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("GPU Time (Warp-Tiled): %f seconds\n",
           std::chrono::duration<double>(end_gpu - start_gpu).count());
    printf("Sample result C[0][0] = %d\n", h_C[0]);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

    return 0;
}
