#include <iostream>
#include <string>
#include <map>
#include <functional>
#include "test_harness.hpp"
#include "benchmark_harness.hpp"
#include "tuning_harness.hpp"
#include "kernel_interface.hpp"

// Kernel function declarations
void singlethread_reference(int M, int N, int K, float* A, float* B, float* C);
void singlethread_reference_setup();
void singlethread_reference_teardown();

void naive_single(int M, int N, int K, float* A, float* B, float* C);
void naive_single_setup();
void naive_single_teardown();

// Map of available kernels
std::map<std::string, KernelInterface> kernels = {
    {"reference", KernelInterface(singlethread_reference, singlethread_reference_setup, singlethread_reference_teardown)},
    {"naive", KernelInterface(naive_single, naive_single_setup, naive_single_teardown)}
};

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <command> [kernel_name] [options]" << std::endl;
    std::cout << "Commands:" << std::endl;
    std::cout << "  test <kernel_name>     - Test kernel correctness" << std::endl;
    std::cout << "  bench <kernel_name>    - Benchmark kernel performance" << std::endl;
    std::cout << "  tune <kernel_name>     - Tune kernel hyperparameters" << std::endl;
    std::cout << "  compare <k1> <k2>      - Compare two kernels" << std::endl;
    std::cout << "  list                   - List available kernels" << std::endl;
    std::cout << "Available kernels:" << std::endl;
    for (const auto& [name, _] : kernels) {
        std::cout << "  " << name << std::endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string command = argv[1];

    if (command == "list") {
        std::cout << "Available kernels:" << std::endl;
        for (const auto& [name, _] : kernels) {
            std::cout << "  " << name << std::endl;
        }
        return 0;
    }

    if (command == "test") {
        if (argc < 3) {
            std::cout << "Usage: " << argv[0] << " test <kernel_name>" << std::endl;
            return 1;
        }

        std::string kernel_name = argv[2];
        if (kernels.find(kernel_name) == kernels.end()) {
            std::cout << "Unknown kernel: " << kernel_name << std::endl;
            return 1;
        }

        bool success = TestHarness::comprehensive_test(kernels[kernel_name]);
        return success ? 0 : 1;
    }

    if (command == "bench") {
        if (argc < 3) {
            std::cout << "Usage: " << argv[0] << " bench <kernel_name> [M] [N] [K] [iterations]" << std::endl;
            return 1;
        }

        std::string kernel_name = argv[2];
        if (kernels.find(kernel_name) == kernels.end()) {
            std::cout << "Unknown kernel: " << kernel_name << std::endl;
            return 1;
        }

        int M = argc > 3 ? std::stoi(argv[3]) : 256;
        int N = argc > 4 ? std::stoi(argv[4]) : 256;
        int K = argc > 5 ? std::stoi(argv[5]) : 256;
        int iterations = argc > 6 ? std::stoi(argv[6]) : BenchmarkConfig::DEFAULT_ITERATIONS;

        auto result = BenchmarkHarness::benchmark_kernel(kernels[kernel_name], M, N, K, iterations);
        result.print(kernel_name);

        return 0;
    }

    if (command == "tune") {
        if (argc < 3) {
            std::cout << "Usage: " << argv[0] << " tune <kernel_name> [M] [N] [K]" << std::endl;
            return 1;
        }

        std::string kernel_name = argv[2];
        if (kernels.find(kernel_name) == kernels.end()) {
            std::cout << "Unknown kernel: " << kernel_name << std::endl;
            return 1;
        }

        int M = argc > 3 ? std::stoi(argv[3]) : 128;
        int N = argc > 4 ? std::stoi(argv[4]) : 128;
        int K = argc > 5 ? std::stoi(argv[5]) : 128;

        std::cout << "Note: Basic tuning framework available, but kernel-specific tuning requires parametrized kernels." << std::endl;
        std::cout << "Testing kernel '" << kernel_name << "' with matrix size " << M << "x" << N << "x" << K << std::endl;

        auto result = BenchmarkHarness::benchmark_kernel(kernels[kernel_name], M, N, K, 50);
        result.print(kernel_name);

        return 0;
    }

    if (command == "compare") {
        if (argc < 4) {
            std::cout << "Usage: " << argv[0] << " compare <kernel1> <kernel2> [M] [N] [K] [iterations]" << std::endl;
            return 1;
        }

        std::string kernel1_name = argv[2];
        std::string kernel2_name = argv[3];

        if (kernels.find(kernel1_name) == kernels.end()) {
            std::cout << "Unknown kernel: " << kernel1_name << std::endl;
            return 1;
        }
        if (kernels.find(kernel2_name) == kernels.end()) {
            std::cout << "Unknown kernel: " << kernel2_name << std::endl;
            return 1;
        }

        int M = argc > 4 ? std::stoi(argv[4]) : 256;
        int N = argc > 5 ? std::stoi(argv[5]) : 256;
        int K = argc > 6 ? std::stoi(argv[6]) : 256;
        int iterations = argc > 7 ? std::stoi(argv[7]) : BenchmarkConfig::DEFAULT_ITERATIONS;

        BenchmarkHarness::compare_kernels(kernels[kernel1_name], kernels[kernel2_name],
                                        kernel1_name, kernel2_name,
                                        M, N, K, iterations);
        return 0;
    }

    print_usage(argv[0]);
    return 1;
}
