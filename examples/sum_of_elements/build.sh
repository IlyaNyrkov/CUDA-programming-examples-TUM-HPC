#!/bin/bash

# Build CUDA files in final_sum_on_cpu
echo "🔧 Building final_sum_on_cpu..."
for file in final_sum_on_cpu/*.cu; do
    filename=$(basename "$file" .cu)
    nvcc "$file" -o "final_sum_on_cpu/$filename"
done

# Build CUDA files in final_sum_on_atomic
echo "🔧 Building final_sum_on_atomic..."
for file in final_sum_on_atomic/*.cu; do
    filename=$(basename "$file" .cu)
    nvcc "$file" -o "final_sum_on_atomic/$filename"
done

echo "✅ Build complete."
