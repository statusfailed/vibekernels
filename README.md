# Vibe Kernels

Fast CPU matrix multiplication kernels built by LLMs.
See PROMPT.md for LLM instructions.

## Kernels

**Reference**: OpenBLAS single-threaded SGEMM ~147.07 GFLOPS

| Kernel | Technique | GFLOPS | vs Reference |
|--------|-----------|--------|--------------|
| blocked                        | Cache blocking            | ~4.96   | 3.4%     |
| blocked_avx2                   | AVX2 SIMD                 | ~25.66  | 17.4%    |
| blocked_avx2_microkernel4x4    | 4×8 microkernel + packing | ~83.40  | 56.7%    |
| blocked_avx512_microkernel4x16 | AVX512 16-wide            | ~162.31 | 110.4%   |
| blocked_avx512_microkernel4x16_prefetch | Memory prefetching        | ~154.52 | 105.1%   |

## Usage

```bash
nix develop
make clean && make

./run list                   # List kernels
./run test <kernel>          # Test correctness  
./run bench <kernel>         # Benchmark performance
./run compare <k1> <k2>      # Compare kernels
```

## Architecture

- `kernels/` - Kernel implementations  
- `*_harness.hpp` - Test/benchmark/tuning frameworks
- `main.cpp` - CLI interface

Each kernel adds one optimization technique for systematic performance analysis.
