# Matmul Kernel Tricks

Task: write a matmul kernel which achieves >= 80% of Openblas GEMM performance.

Start with some initial ground work. Implement:

- Implement the *reference kernel* using openblas `cblas_sgemm`.
    - It should use *ONE THREAD*.
    - Name it `singlethread_reference.cpp`
- Implement a *test harness* which checks correctness against the naive implementation
- Write a *tuning harness* which selects the best hyperparameters for a given kernel
- Implement a *benchmark harness* which checks performance over a number of iterations, recording mean, min, max perf.
- Benchmark the reference kernel, and keep the static data on its

Follow this loop:

- Write a new kernel
- Put it in its own file, using a descriptive name (name it after *techniques* used, e.g. `singlethread_blocked_microkernel_AxB.cpp`)
- Test the kernel
- Tune the kernel to this hardware
- Run the kernel
- If performance < 80% of reference, loop.


Note:

- In more advanced kernels, you are allowed to use special instructions on CPU (e.g., SIMD)
- Keep code modular: each kernel should be a self contained file
- Re-use the testing/performance/tuning harnesses for every kernel.
    - You may need to design an interface to do this for tuning.

## Multicore

Once we hit 80% of single threaded perf, we'll move onto multi-core, and repeat.
That process will look like this:

- Implement `multithreaded_reference.cpp` (should be cblass_sgemm with thread params set)
- Ensure we get consistent performance between runs of the reference kernel
    - Investigate any sources of performance variance
- Follow the loop above
