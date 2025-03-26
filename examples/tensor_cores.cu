#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cublas_v2.h>


// A macro to check for errors in cuBLAS function calls. 
// If a cuBLAS function fails, it prints an error message and exits the program.
#define CHECK_CUBLAS(call)                                                   \
    {                                                                        \
        cublasStatus_t err = call;                                           \
        if (err != CUBLAS_STATUS_SUCCESS) {                                  \
            printf("CUBLAS error %d at %s:%d\n", err, __FILE__, __LINE__);   \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    }

void fill_matrix(half *matrix, int rows, int cols, float value) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = __float2half(value); // Fill with half-precision values
    }
}

int main() {
    const int M = 512; // Rows of A and C
    const int N = 512; // Columns of B and C
    const int K = 512; // Columns of A, Rows of B

    // Host and device matrices
    half *h_A, *h_B;
    float *h_C;

    half *d_A, *d_B;
    float *d_C;

    size_t size_A = M * K * sizeof(half);
    size_t size_B = K * N * sizeof(half);
    size_t size_C = M * N * sizeof(float);

    // Allocate host memory
    h_A = (half *)malloc(size_A);
    h_B = (half *)malloc(size_B);
    h_C = (float *)malloc(size_C);

    // Initialize host matrices
    fill_matrix(h_A, M, K, 1.0f);
    fill_matrix(h_B, K, N, 1.0f);
    for (int i = 0; i < M * N; i++) h_C[i] = 0.0f;

    // Allocate device memory
    cudaMalloc((void **)&d_A, size_A);
    cudaMalloc((void **)&d_B, size_B);
    cudaMalloc((void **)&d_C, size_C);

    // Copy data to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);

    // cuBLAS handle
    cublasHandle_t handle;
    // Creates a cuBLAS handle, which is required for all cuBLAS operations.
    CHECK_CUBLAS(cublasCreate(&handle));

    // Alpha and beta values for the operation
    float alpha = 1.0f;
    float beta = 0.0f;

    // Tensor Core GEMM
    // cublasGemmEx performs the generalized matrix multiplication C=α(A×B)+βC.

    /*
CUBLAS_OP_N: Indicates no transpose for matrices  A and B.
M, N, K: Dimensions of the matrices.
&alpha and &beta: Scalars for the computation.
d_A, d_B, d_C: Pointers to matrices on the GPU.
CUDA_R_16F: Specifies that A and B are in FP16 format.
CUDA_R_32F: Specifies that C is in FP32 format.
CUBLAS_GEMM_DEFAULT_TENSOR_OP: Enables Tensor Core acceleration.
    */
    CHECK_CUBLAS(cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        M, N, K,
        &alpha,
        d_A, CUDA_R_16F, M,
        d_B, CUDA_R_16F, K,
        &beta,
        d_C, CUDA_R_32F, M,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Print some results
    printf("C[0]: %f\n", h_C[0]); // Should be 512.0f (sum of 512 elements)
    printf("C[M*N-1]: %f\n", h_C[M * N - 1]); // Should also be 512.0f

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    return 0;
}