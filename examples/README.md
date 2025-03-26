# CUDA-programming-examples-TUM-HPC
## Code examples

### Simple cuda threads program

Build cuda program:

```bash
nvcc print_kernel.cu -o print_kernel
```

Run cuda program:

```bash
./print_kernel
```

Expected result: 
```shell
block id: 1, thread id: 0
block id: 1, thread id: 1
block id: 1, thread id: 2
block id: 1, thread id: 3
block id: 0, thread id: 0
block id: 0, thread id: 1
block id: 0, thread id: 2
block id: 0, thread id: 3
```

