#include <stdio.h>
#include <iostream>

__global__ void deadlockExample(){
  __shared__ int semaphore;
  semaphore=0;
  __syncthreads();
  while (true) {
    int prev=atomicCAS(&semaphore,0,1);
    if (prev==0) {
      //critical section
      semaphore=0;
      break;
    }
  }
}


int main(){
    const int BLOCK_COUNT = 1;
    const int THREAD_COUNT = 2;

    deadlockExample<<<BLOCK_COUNT, THREAD_COUNT>>>();
    cudaDeviceSynchronize();
    std::cout << "Program finished" << std::endl;
}