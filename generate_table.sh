#!/usr/bin/env bash

# Script to benchmark all kernels and generate performance table
# Usage: ./generate_table.sh

set -e

echo "Benchmarking kernels..."

declare -A techniques=(
    ["naive"]="Triple loops"
    ["blocked"]="Cache blocking"
    ["blocked_avx2"]="AVX2 SIMD"
    ["blocked_avx2_microkernel4x4"]="4×8 microkernel + packing"
    ["blocked_avx512_microkernel4x16"]="AVX512 16-wide"
    ["blocked_avx512_microkernel4x16_prefetch"]="Memory prefetching"
)

declare -A results=()

# Get reference performance
echo "Getting reference performance..."
ref_output=$(nix develop --command bash -c "./run bench reference 2>/dev/null")
ref_gflops=$(echo "$ref_output" | grep "Mean:" | awk '{print $2}')
echo "Reference: $ref_gflops GFLOPS"

echo ""
echo "| Kernel | Technique | GFLOPS | vs Reference |"
echo "|--------|-----------|--------|--------------|"

# Benchmark each kernel
for kernel in blocked blocked_avx2 blocked_avx2_microkernel4x4 blocked_avx512_microkernel4x16 blocked_avx512_microkernel4x16_prefetch; do
    output=$(nix develop --command bash -c "./run bench $kernel 2>/dev/null" || echo "FAILED")
    
    if [[ "$output" == "FAILED" ]]; then
        echo "FAILED"
        gflops="N/A"
        percent="N/A"
    else
        gflops=$(echo "$output" | grep "Mean:" | awk '{print $2}')
        percent=$(awk "BEGIN {printf \"%.1f\", $gflops * 100 / $ref_gflops}")
        #echo "$gflops GFLOPS"
    fi
    
    technique="${techniques[$kernel]}"
    printf "| %-30s | %-25s | ~%-6s | %-8s |\n" "$kernel" "$technique" "$gflops" "${percent}%"
done

echo ""
echo "**Reference**: OpenBLAS single-threaded SGEMM ~$ref_gflops GFLOPS"
