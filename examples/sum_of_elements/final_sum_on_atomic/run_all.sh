#!/bin/bash

# Folder with compiled binaries
BIN_DIR="final_sum_on_cpu"
N=${1:-4194304}  # Default to 2^22

cd "$BIN_DIR" || exit 1

echo "Running all benchmarks with N = $N"

for binary in *; do
    # Skip .cu files
    [[ "$binary" == *.cu ]] && continue

    if [[ -x "$binary" ]]; then
        echo -e "\n▶️ Running: $binary"
        ./"$binary" "$N"
    fi
done

echo -e "\n✅ All runs completed."
