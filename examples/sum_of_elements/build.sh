#!/bin/bash

# replace for your architecture -arch=sm_90 
# Build CUDA files in final_sum_on_cpu
echo "🔧 Building final_sum_on_cpu..."
for file in final_sum_on_cpu/*.cu; do
    filename=$(basename "$file" .cu)
    nvcc "$file" -O3 -arch=sm_90 -o "final_sum_on_cpu/$filename"
done

# Build CUDA files in final_sum_on_atomic
echo "🔧 Building final_sum_on_atomic..."
for file in final_sum_on_atomic/*.cu; do
    filename=$(basename "$file" .cu)
    nvcc "$file" -O3 -arch=sm_90 -o "final_sum_on_atomic/$filename"
done

# Build CUDA files in cudaMalloc
echo "🔧 Building cudaMalloc..."
for file in cudaMalloc/*.cu; do
    filename=$(basename "$file" .cu)
    nvcc "$file" -O3 -arch=sm_90 -o "cudaMalloc/$filename"
done

echo "✅ Build complete."
