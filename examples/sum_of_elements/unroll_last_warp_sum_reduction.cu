#include <stdio.h>
#include <time.h>
#include <iostream>
#include <chrono>

using namespace std;

void fill_array(int N, int max_val, int* x) {
    srand(time(0));
    for (int i = 0; i < N; i++) {
        x[i] = rand() % (max_val + 1);
    }
}

void print_array(int N, int* x) {
    printf("printing first %d elements", N);
    for (int i = 0; i < N; i++) {
        printf("%d ", x[i]);
    }

    printf("\n");
}


void sumCPU(int *in, int *out, int n) {
    for (int i = 0; i < n; i++) {
        (*out) += in[i];
    }
}

__device__ void warpReduceUnrolled(volatile int* s, int tid) {
    s[tid] += s[tid + 32];
    s[tid] += s[tid + 16];
    s[tid] += s[tid + 8];
    s[tid] += s[tid + 4];
    s[tid] += s[tid + 2];
    s[tid] += s[tid + 1];
}

__global__ void unrollLastWarpReduction(int *in, int *out, int n) {
    extern __shared__ int subArray[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    int val1 = (gid < n) ? in[gid] : 0;
    int val2 = (gid + blockDim.x < n) ? in[gid + blockDim.x] : 0;

    subArray[tid] = val1 + val2;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 32; stride /= 2) {
        if (tid < stride) {
            subArray[tid] += subArray[tid + stride];
        }
        __syncthreads();
    }

    if (tid < 32) {
        warpReduceUnrolled(subArray, tid);
    }

    if (tid == 0) {
        atomicAdd(out, subArray[0]);
    }
}


int main() {
    const int N = 1 << 10;
    const int max = 100;

    int* in, out_cpu, out_gpu;
    cudaMallocManaged(&in, N*sizeof(int));
    fill_array(N, 100, in);
    print_array(10, in);

    auto start_cpu = chrono::high_resolution_clock::now();
    sumCPU(in, out_cpu, N);
    auto end_cpu = chrono::high_resolution_clock::now();

    chrono::duration<double> cpu_time = end_cpu - start_cpu;

    cout << "CPU Array sum time: " << cpu_time.count() << " seconds" << endl;
    cout << "CPU Array sum result: " << *out_cpu << endl; 

    const int THREAD_COUNT = 128;
    const int BLOCK_COUNT = (N + (THREAD_COUNT - 1)) / THREAD_COUNT;

    auto start_gpu = chrono::high_resolution_clock::now();
    unrollLastWarpReduction<<<BLOCK_COUNT, THREAD_COUNT, THREAD_COUNT>>>(in, out_gpu, N);
    cudaDeviceSynchronize();
    auto end_gpu = chrono::high_resolution_clock::now();

    chrono::duration<double> gpu_time = end_gpu - start_gpu;

    cout << "GPU Array sum time: " << gpu_time.count() << " seconds" << endl;
    cout << "GPU Array sum result: " << *out_gpu << endl; 


}