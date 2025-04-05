#!/bin/bash

# Default number of elements is 2^22 if not specified
N=${1:-4194304}

echo "🚀 Running all binaries with N = $N elements"

# Run binaries in final_sum_on_cpu
echo "🔹 Running binaries in final_sum_on_cpu:"
for bin in final_sum_on_cpu/*; do
    if [[ -x "$bin" && ! "$bin" =~ \.cu$ ]]; then
        echo "▶️  $bin"
        "$bin" "$N"
        echo "-----------------------------------------"
    fi
done

# Run binaries in final_sum_on_atomic
echo "🔹 Running binaries in final_sum_on_atomic:"
for bin in inal_sum_on_atomic/*; do
    if [[ -x "$bin" && ! "$bin" =~ \.cu$ ]]; then
        echo "▶️  $bin"
        "$bin" "$N"
        echo "-----------------------------------------"
    fi
done
