#include <stdio.h>

__global__ void hello(){
    printf("block id: %d, thread id: %d\n", blockIdx.x, threadIdx.x);
}

int main(){
    // Launch the kernel
    // first - blocks, second - threads per block
    const int BLOCK_COUNT = 4;
    const int THREAD_COUNT = 16;

    hello<<<BLOCK_COUNT, THREAD_COUNT>>>();
    cudaDeviceSynchronize();

    return 0;
}