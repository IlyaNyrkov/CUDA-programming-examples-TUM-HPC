# CUDA-programming-examples-TUM-HPC

![CUDA][cuda-shield] ![C++][cpp-shield] ![Linux][linux-shield]

This repository contains a collection of CUDA programs developed for an educational article and the TUM High Performance Computing (HPC) course. The examples demonstrate core CUDA concepts, performance optimization techniques, and GPU-accelerated computation through small, focused programs.

## Structure

- `examples/` — All core CUDA examples grouped by topic (reduction, Riemann sum, matrix ops, deadlocks, etc.)
- `README.md` — You are here.
- Each folder contains a self-contained `*.cu` example with comments.

## Topics Covered

- CUDA thread blocks and kernels
- Deadlock using atomics and synchronization
- Sum reduction (with multiple optimization levels)
- Matrix multiplication (including Tensor Core version)
- Riemann sum integration using different parallelization techniques
- Shared vs. global memory usage
- Unified memory
- Warp-level primitives
- Performance comparison (CPU vs. GPU)

---

## 🛠️ Compiler Options

Some programs require special flags:

### For double-precision atomic operations (used in `riemann_sum/*.cu`):

```bash
nvcc --compiler-args=-arch=sm_60 your_file.cu -o your_program
```

### For kernel invocation within other kernel (quicksort example)

```bash
nvcc --compiler-args=-rdc=true your_file.cu -o your_program
```