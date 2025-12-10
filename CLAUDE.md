# Matrix Multiplication Optimization Project

## Project Overview
This is a C++ matrix multiplication optimization framework for experimenting with different kernel implementations and measuring their performance against OpenBLAS reference.

## Project Structure
```
.
├── main.cpp                    # CLI interface (renamed binary to 'run')
├── Makefile                   # Build system with OpenBLAS + OpenMP
├── kernel_interface.hpp       # Kernel interface with setup/teardown support
├── test_harness.hpp          # Correctness testing framework
├── benchmark_harness.hpp     # Performance measurement framework
├── tuning_harness.hpp        # Hyperparameter optimization framework
└── kernels/
    ├── singlethread_reference.cpp  # OpenBLAS reference (properly single-threaded)
    └── naive_single.cpp            # Naive triple-loop implementation
```

## Current Implementation Status

### Completed Components
- ✅ **KernelInterface**: Support for setup/teardown functions outside hot loop
- ✅ **Single-threaded Reference**: Properly constrained to ~107 GFLOPS using both `omp_set_num_threads(1)` and `openblas_set_num_threads(1)`
- ✅ **Naive Implementation**: Basic triple-loop baseline
- ✅ **Test Harness**: Correctness verification against reference
- ✅ **Benchmark Harness**: Statistical performance measurement (mean, median, std dev)
- ✅ **Tuning Harness**: Framework for hyperparameter optimization
- ✅ **CLI Interface**: Commands for test, bench, tune, list operations

### Performance Baselines
- **Reference (OpenBLAS single-thread)**: ~107 GFLOPS (properly constrained)
- **Naive implementation**: ~0.5-1 GFLOPS baseline

## Critical Technical Details

### Threading Configuration
⚠️ **IMPORTANT**: Achieving true single-threaded performance required BOTH:
1. `omp_set_num_threads(1)` - Controls OpenMP threading
2. `openblas_set_num_threads(1)` - Controls OpenBLAS threading

Using only one of these resulted in multi-threaded execution (~400+ GFLOPS instead of ~100 GFLOPS).

### Build System
- **Dependencies**: OpenBLAS (`-lopenblas`) + OpenMP (`-lgomp`)
- **Environment**: Nix development shell with openblas available
- **Compiler Flags**: `-O3 -std=c++17 -Wall -Wextra`
- **Note**: `-march=native` disabled in Nix environment

### Kernel Interface Architecture
```cpp
struct KernelInterface {
    void (*kernel)(int, int, int, float*, float*, float*);
    void (*setup)();     // Called once before benchmarking
    void (*teardown)();  // Called once after benchmarking
};
```

This allows expensive setup operations (thread configuration, memory allocation) to be performed outside the performance measurement hot loop.

## Usage Commands

```bash
# Build the project
make clean && make

# List available kernels
./run list

# Test kernel correctness
./run test <kernel_name>

# Benchmark kernel performance
./run bench <kernel_name>

# Tune kernel hyperparameters
./run tune <kernel_name>

# Convenience targets
make bench-reference    # Quick reference benchmark
make test-reference     # Quick reference test
```

## Development Notes

### Adding New Kernels
1. Create `kernels/your_kernel.cpp` with implementation
2. Include setup/teardown functions if needed for configuration
3. Add to kernel registry in `main.cpp`
4. Test correctness with `./run test your_kernel`
5. Benchmark with `./run bench your_kernel`

### Performance Testing
- Default benchmark: 256x256x256 matrices, 100 iterations
- Includes warm-up runs to reduce measurement noise
- Reports mean, median, min, max, and standard deviation
- All measurements in GFLOPS (billions of floating-point operations per second)

### Known Working Environment
- Nix development shell with OpenBLAS package
- GCC 14.3.0 compiler
- Linux environment
- Architecture: x86_64

## Next Steps for Optimization
1. **Cache-aware implementations**: Blocking/tiling strategies
2. **SIMD optimizations**: AVX/AVX2 vector instructions  
3. **Multi-threading**: Properly parallelized kernels
4. **Memory prefetching**: Reduce cache misses
5. **Mixed precision**: fp16/bf16 optimizations

## Important Lint/Build Commands
- `make clean && make` - Full rebuild
- Always run tests before benchmarking new implementations
- Use `./run list` to verify kernel registration

## Project Goals
Achieve competitive performance against OpenBLAS while maintaining code clarity and providing educational value for understanding matrix multiplication optimization techniques.