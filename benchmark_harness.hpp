#pragma once
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <cmath>

struct BenchmarkResult {
    double mean_gflops;
    double min_gflops;
    double max_gflops;
    double std_gflops;
    double median_gflops;
    std::vector<double> all_measurements;
    
    void print(const std::string& kernel_name) const {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "\n=== Benchmark Results for " << kernel_name << " ===" << std::endl;
        std::cout << "Mean:   " << std::setw(8) << mean_gflops << " GFLOPS" << std::endl;
        std::cout << "Min:    " << std::setw(8) << min_gflops << " GFLOPS" << std::endl;
        std::cout << "Max:    " << std::setw(8) << max_gflops << " GFLOPS" << std::endl;
        std::cout << "Median: " << std::setw(8) << median_gflops << " GFLOPS" << std::endl;
        std::cout << "StdDev: " << std::setw(8) << std_gflops << " GFLOPS" << std::endl;
        std::cout << "Samples: " << all_measurements.size() << std::endl;
        std::cout << "============================================" << std::endl;
    }
};

class BenchmarkHarness {
public:
    static BenchmarkResult benchmark_kernel(void (*kernel)(int, int, int, float*, float*, float*),
                                           int M, int N, int K, int iterations = 100) {
        std::vector<float> A(M * K);
        std::vector<float> B(K * N);
        std::vector<float> C(M * N);
        
        // Initialize with random-ish values to prevent compiler optimizations
        for (int i = 0; i < M * K; i++) A[i] = static_cast<float>(i % 1000) / 1000.0f;
        for (int i = 0; i < K * N; i++) B[i] = static_cast<float>(i % 1000) / 1000.0f;
        
        std::vector<double> measurements;
        measurements.reserve(iterations);
        
        // Warm-up runs
        for (int i = 0; i < 10; i++) {
            std::fill(C.begin(), C.end(), 0.0f);
            kernel(M, N, K, A.data(), B.data(), C.data());
        }
        
        // Actual measurements
        for (int i = 0; i < iterations; i++) {
            std::fill(C.begin(), C.end(), 0.0f);
            
            auto start = std::chrono::high_resolution_clock::now();
            kernel(M, N, K, A.data(), B.data(), C.data());
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            double seconds = duration * 1e-9;
            double operations = 2.0 * M * N * K; // FMA operations
            double gflops = operations / seconds * 1e-9;
            
            measurements.push_back(gflops);
        }
        
        return analyze_measurements(measurements);
    }
    
    static BenchmarkResult benchmark_multiple_sizes(void (*kernel)(int, int, int, float*, float*, float*),
                                                   const std::vector<std::tuple<int, int, int>>& sizes,
                                                   int iterations_per_size = 50) {
        std::vector<double> all_measurements;
        
        std::cout << "Benchmarking across multiple matrix sizes..." << std::endl;
        for (const auto& [M, N, K] : sizes) {
            auto result = benchmark_kernel(kernel, M, N, K, iterations_per_size);
            all_measurements.insert(all_measurements.end(), 
                                  result.all_measurements.begin(), 
                                  result.all_measurements.end());
            
            std::cout << M << "x" << N << "x" << K << ": " 
                     << std::fixed << std::setprecision(2) 
                     << result.mean_gflops << " ± " << result.std_gflops << " GFLOPS" << std::endl;
        }
        
        return analyze_measurements(all_measurements);
    }
    
    static double compare_kernels(void (*kernel1)(int, int, int, float*, float*, float*),
                                void (*kernel2)(int, int, int, float*, float*, float*),
                                const std::string& name1, const std::string& name2,
                                int M = 256, int N = 256, int K = 256, int iterations = 100) {
        auto result1 = benchmark_kernel(kernel1, M, N, K, iterations);
        auto result2 = benchmark_kernel(kernel2, M, N, K, iterations);
        
        result1.print(name1);
        result2.print(name2);
        
        double speedup = result1.mean_gflops / result2.mean_gflops;
        double percentage = (result1.mean_gflops / result2.mean_gflops) * 100.0;
        
        std::cout << "\nComparison:" << std::endl;
        std::cout << name1 << " vs " << name2 << ": " 
                 << std::fixed << std::setprecision(2) 
                 << speedup << "x speedup (" << percentage << "% of " << name2 << " performance)" << std::endl;
        
        return percentage;
    }

private:
    static BenchmarkResult analyze_measurements(const std::vector<double>& measurements) {
        BenchmarkResult result;
        result.all_measurements = measurements;
        
        // Calculate statistics
        result.mean_gflops = std::accumulate(measurements.begin(), measurements.end(), 0.0) / measurements.size();
        result.min_gflops = *std::min_element(measurements.begin(), measurements.end());
        result.max_gflops = *std::max_element(measurements.begin(), measurements.end());
        
        // Median
        std::vector<double> sorted_measurements = measurements;
        std::sort(sorted_measurements.begin(), sorted_measurements.end());
        size_t n = sorted_measurements.size();
        result.median_gflops = (n % 2 == 0) ? 
            (sorted_measurements[n/2-1] + sorted_measurements[n/2]) / 2.0 :
            sorted_measurements[n/2];
        
        // Standard deviation
        double variance = 0.0;
        for (double measurement : measurements) {
            variance += (measurement - result.mean_gflops) * (measurement - result.mean_gflops);
        }
        variance /= measurements.size();
        result.std_gflops = std::sqrt(variance);
        
        return result;
    }
};