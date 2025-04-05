#!/bin/bash

# Folder with CUDA source files
SRC_DIR="final_sum_on_cpu"

# Navigate into the folder
cd "$SRC_DIR" || exit 1

# Compile each .cu file using nvcc
for cu_file in *.cu; do
    exe_name="${cu_file%.cu}"
    echo "Compiling $cu_file -> $exe_name"
    nvcc -O3 "$cu_file" -o "$exe_name"
done

echo "✅ All files compiled."
