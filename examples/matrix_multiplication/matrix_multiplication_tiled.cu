#include <stdio.h>
#include <cuda.h>
#include <chrono>

#define TILE_SIZE 16                // L40S: 16–32, A100/H100: 32 works well
#define BLOCK_SIZE TILE_SIZE

__global__ void tiledMatMul(int* A, int* B, int* C, int N) {
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

void fill_matrix(int *mat, int N) {
    for (int i = 0; i < N * N; ++i) {
        mat[i] = rand() % 10;
    }
}

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : 512;
    size_t size = N * N * sizeof(int);

    int *A, *B, *C;
    cudaMallocHost(&A, size);
    cudaMallocHost(&B, size);
    cudaMallocHost(&C, size);

    fill_matrix(A, N);
    fill_matrix(B, N);

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    auto start_gpu = std::chrono::high_resolution_clock::now();
    tiledMatMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    printf("Tiled GPU Matrix Multiplication (%d x %d)\n", N, N);
    printf("GPU Execution Time: %f seconds\n",
           std::chrono::duration<double>(end_gpu - start_gpu).count());
    printf("Sample result C[0][0] = %d\n", C[0]);

    // Cleanup
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
