#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include "time.h"
#include <cstdio>
#include <chrono>
using namespace std;

void swap(int* a, int* b) {
    int c = *a;
    *a = *b;
    *b = c;
}

int quick_sort_partition(int* x, int l, int r) {
    int pivot_val = x[l];
    int left = l;
    
    for (int i = l + 1; i < r; i++) {
        if (x[i] < pivot_val) {
            swap(&x[i], &x[left]);
            left++;
        }
    }
    
    x[left] = pivot_val;
    
    return left;
}

void quick_sort(int* x, int l, int r) {
    if (l < r) {
        int pivot_id = quick_sort_partition(x, l, r);
        quick_sort(x, l, pivot_id);
        quick_sort(x, pivot_id + 1, r);
    }
}


__device__ void gpu_swap(int* a, int* b) {
    int c = *a;
    *a = *b;
    *b = c;
}

__device__ int quick_sort_partition_gpu(int *x, int l, int r) {
    int pivot_val = x[l];
    int left = l;

    for (int i = l + 1; i < r; i++) {
        if (x[i] < pivot_val) {
            gpu_swap(&x[i], &x[left]);
            left++;
        }
    }

    x[left] = pivot_val;

    return left;

}


__global__ void quick_sort_gpu(int* x, int l, int r) {
    if (l < r) {
        int pivot_id = quick_sort_partition_gpu(x, l, r);
        quick_sort_gpu<<<1, 1>>>(x, l, pivot_id);
        quick_sort_gpu<<<1, 1>>>(x, pivot_id + 1, r);
    }
}

// helper functions

void print_array(int *x, int N) {
    for (int i = 0; i < N; i++) {
        printf("%d ", x[i]);
    }

    printf("\n");
}

void fill_array(int *x, int N, int max) {
    for (int i = 0; i < N; i++) {
        x[i] = rand() % max;
    }
}

void copy_array(int *original, int *copy, int N) {
   for (int i = 0; i < N; i++) {
       copy[i] = original[i];
   }
}

int compare_arrays(int *x, int *y, int N) {
   for (int i = 0; i < N; i++) {
      if (x[i] != y[i]) {
          return 0;
      }
   }

   return 1;
}

int main() {
    srand(time(0));
    int *x;
    const int N = 100000000;
    
    cout << "Elements in array N = " << N << endl;

    cudaMallocManaged(&x, N * sizeof(int));
    fill_array(x, N, N);
    print_array(x, 10);
    
    int *y;
    cudaMallocManaged(&y, N * sizeof(int));
    copy_array(x, y, N);
    print_array(y, 10);

    auto start_gpu = chrono::high_resolution_clock::now();

    quick_sort(x, 0, N);

    auto end_gpu = chrono::high_resolution_clock::now();
    chrono::duration<double> cpu_time = end_gpu - start_gpu;
    cout << "\n Quick sort on cpu execution time " << cpu_time.count() << " seconds" << endl;
    print_array(x, 10);

    start_gpu = chrono::high_resolution_clock::now();
    
    quick_sort_gpu<<<1, 1>>>(y, 0, N);
    cudaDeviceSynchronize();

    end_gpu = chrono::high_resolution_clock::now();
    chrono::duration<double> gpu_time = end_gpu - start_gpu;

    print_array(y, 10);
    cout << "\n Quick sort on GPU execution time " << gpu_time.count() << " seconds" << endl;

    cout <<  "array and matrix quicksort are the same?: " <<  compare_arrays(x, y, N) << endl;    

    return 0;
}
