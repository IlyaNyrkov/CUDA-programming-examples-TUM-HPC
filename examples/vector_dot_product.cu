#include <stdio.h>

__global__ void vectorDotProduct(float3 *a, float3 *b, float *result, int n) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n) {
        // Perform element-wise dot product
        float3 va = a[id];
        float3 vb = b[id];
        result[id] = va.x * vb.x + va.y * vb.y + va.z * vb.z;
    }
}

int main() {
    int N = 1024;
    float3 *a, *b;
    float *result;

    cudaMallocManaged(&a, N * sizeof(float3));
    cudaMallocManaged(&b, N * sizeof(float3));
    cudaMallocManaged(&result, N * sizeof(float));

    // Initialize vectors
    for (int i = 0; i < N; i++) {
        a[i] = make_float3(1.0f, 2.0f, 3.0f);
        b[i] = make_float3(4.0f, 5.0f, 6.0f);
    }

    // Launch kernel
    vectorDotProduct<<<(N + 255) / 256, 256>>>(a, b, result, N);
    cudaDeviceSynchronize();

    // Print results
    for (int i = 0; i < 5; i++) {
        printf("Dot product of vector %d: %f\n", i, result[i]);
    }

    // Free memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(result);

    return 0;
}
