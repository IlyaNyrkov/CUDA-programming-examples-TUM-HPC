// cuBLAS handle
cublasHandle_t handle;
// Creates a cuBLAS handle,
// which is required for 
// all cuBLAS operations
CHECK_CUBLAS(cublasCreate(&handle));
// Alpha and beta coefficients
float alpha = 1.0f;
float beta = 0.0f;
// Tensor Core GEMM
// cublasGemmEx performs the 
// generalized matrix multiplication
// C=alpha*(A*B)+beta*C  
// FMA operation with coefficients
/*
CUBLAS_OP_N: Indicates no transpose 
for matrices A and B.
M, N, K: Dimensions of the matrices.
alpha,beta: Scalars for the computation.
d_A,d_B,d_C: Pointers to matrices.
CUDA_R_16F: A/B are in FP16 format.
CUDA_R_32F: C is in FP32 format.
CUBLAS_GEMM_DEFAULT_TENSOR_OP: 
Enables Tensor Core acceleration.
*/
CHECK_CUBLAS(cublasGemmEx(
    handle, CUBLAS_OP_N, CUBLAS_OP_N,
    M, N, K,
    &alpha,
    d_A, CUDA_R_16F, M,
    d_B, CUDA_R_16F, K,
    &beta,
    d_C, CUDA_R_32F, M,
    CUDA_R_32F,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP));