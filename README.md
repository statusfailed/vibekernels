# Matmul Kernel Tricks

Task: write a matmul kernel which achieves >= 80% of Openblas GEMM performance.

Start with some initial ground work. Implement:

- Implement a *single-threaded reference kernel* using openblas `cblas_sgemm`. Straight matmul, ignore alpha etc.
- Implement a *test harness* which checks correctness against the naive implementation
- Write a *tuning harness* which selects the best hyperparameters for a given kernel
- Implement a *benchmark harness* which checks performance over a number of iterations, recording mean, min, max perf.
- Write a `main.cpp` and a Makefile for the project; ensure all kernels can be run by e.g. `./bench <kernel_name>`
- Benchmark the reference kernel, and keep the static data on its performance
- Collect information on this machine (e.g., cat /proc/cpuinfo), and use it to
  write down a prioritised list of possible performance improvements to the naive
  kernel (e.g., SIMD, blocking, microkernel, etc.)

Follow this loop:

- Write a new kernel
- Put it in its own file, using a descriptive name (name it after *techniques* used, e.g. `singlethread_blocked_microkernel_AxB.cpp`)
- Test the kernel
- Tune the kernel to this hardware
- Run the kernel
- If performance < 80% of reference, loop.


Notes:

- In more advanced kernels, you are allowed to use special instructions on CPU (e.g., SIMD)
- Re-use the testing/performance/tuning harnesses for every kernel.
    - You may need to design an interface to do this for tuning.
- Nix flakes are used for dependencies. See `flake.nix`.
- Keep code modular: each kernel should be a self contained file
    - name the singlethreaded reference kernel `singlethread_reference.cpp`
    - name other kernels following the same convention, e.g. `singlethread_naive.cpp`
    - Kernels should be placed in a `kernels/` dir

## Multicore

Once we hit 80% of single threaded perf, we'll move onto multi-core, and repeat.
That process will look like this:

- Implement `multithreaded_reference.cpp` (should be cblass_sgemm with thread params set)
- Ensure we get consistent performance between runs of the reference kernel
    - Investigate any sources of performance variance
- Follow the loop above
