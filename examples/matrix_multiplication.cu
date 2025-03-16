#include <cstdio>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

// Function to fill a dynamically allocated 2D array with random values
void fill_rand_flat_matrix(int* a, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            a[i * cols + j] = rand() % 10 + 1; // Fill with random values
        }
    }
}

// CUDA kernel for matrix multiplication
__global__ void matrix_multiply_cuda(int* a, int* b, int* c, int rows1, int cols1, int cols2) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows1 && col < cols2) {
        int sum = 0;
        for (int k = 0; k < cols1; k++) {
            sum += a[row * cols1 + k] * b[k * cols2 + col];
        }
        c[row * cols2 + col] = sum;
    }
}

int main() {
    int rows1 = 4000, cols1 = 3000;  // Dimensions of matrix A
    int rows2 = 3000, cols2 = 4000;  // Dimensions of matrix B

    // Flat matricies for CUDA
    int* d_a;
    int* d_b;
    int* d_c;
    int size_a = rows1 * cols1 * sizeof(int);
    int size_b = rows2 * cols2 * sizeof(int);
    int size_c = rows1 * cols2 * sizeof(int);

    // Allocate memory in gpu (bytes)
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    // Flat arrays in RAM (used in future for copying to cuda memory)
    int* flat_a = new int[rows1 * cols1];
    fill_rand_flat_matrix(flat_a, rows1, cols1);
    int* flat_b = new int[rows2 * cols2];
    fill_rand_flat_matrix(flat_b, rows2, cols2);

    cudaMemcpy(d_a, flat_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, flat_b, size_b, cudaMemcpyHostToDevice);

    // Configure CUDA kernel
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows1 + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // capture start time
    auto start_gpu = chrono::high_resolution_clock::now();

    // run kernel for multiplication
    matrix_multiply_cuda<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, rows1, cols1, cols2);
    // cpu waits until gpu finishes with the kernel
    cudaDeviceSynchronize();
    // capture end time
    auto end_gpu = chrono::high_resolution_clock::now();
    chrono::duration<double> gpu_time = end_gpu - start_gpu;

    // allocate memory to copy from gpu to cpu to display result.
    int* flat_c = new int[rows1 * cols2];
    cudaMemcpy(flat_c, d_c, size_c, cudaMemcpyDeviceToHost);

    cout << "\nGPU Matrix Multiplication Time: " << gpu_time.count() << " seconds" << endl;

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] flat_a;
    delete[] flat_b;
    delete[] flat_c;

    return 0;
}
