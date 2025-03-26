# CUDA-programming-examples-TUM-HPC
## Code Examples

This folder contains various CUDA code examples structured by topic. Each example demonstrates specific CUDA programming concepts and GPU performance optimization techniques.

---

### `print_kernel.cu`

Minimal example showing how to launch a CUDA kernel and print thread/block indices.

**Build:**
```bash
nvcc print_kernel.cu -o print_kernel
```

**Run:**
```bash
./print_kernel
```

### ➕ sum_of_elements/
Different implementations of sum reduction, from naive to highly optimized.

* naive_sum_reduction.cu – Each thread uses atomicAdd (slowest).

* block_sum_reduction.cu – Shared memory + atomicAdd at block level.

* first_add_global_load_sum_reduction.cu – Optimizes memory load by summing two elements during load.

* grid_stride_sum_reduction.cu – Uses grid-stride loops for large arrays.

* unroll_last_warp_sum_reduction.cu – Warp unrolling optimization.

* unroll_wrap_completely_sum_reduction.cu – Fully unrolled warp + template-based compile-time optimization.

Each file demonstrates a different step toward high-performance reduction.

### ∫ riemann_sum/
Implements Left Riemann Sum integration using various GPU strategies. Each kernel approximates integrals like:
$$\int_a^b f(x) \,dx$$

* naive_riemann_sum.cu – AtomicAdd per thread.

* grid_stride_riemann_sum.cu – Grid-stride loop pattern.

* loop_unrolling_riemann_sum.cu – Manual loop unrolling.

* warp_unrolling_riemann_sum.cu – Warp shuffle instructions.

* shared_memory_riemann_sum.cu – Shared memory reduction per block.

⚠ Important: These kernels use atomicAdd(double*, double) — make sure to compile with:

```bash
nvcc --compiler-args=-arch=sm_60 file.cu -o program
```
### ✴ matrix_multiplication.cu
Matrix multiplication using shared memory and tiling for better memory coalescing and reuse.
