#pragma once
#include <vector>
#include <chrono>
#include <algorithm>
#include <functional>
#include <iostream>
#include <map>

struct TuningParams {
    std::map<std::string, int> params;
    
    TuningParams& set(const std::string& name, int value) {
        params[name] = value;
        return *this;
    }
    
    int get(const std::string& name, int default_value = 0) const {
        auto it = params.find(name);
        return (it != params.end()) ? it->second : default_value;
    }
};

struct TuningResult {
    TuningParams params;
    double performance_gflops;
    
    bool operator<(const TuningResult& other) const {
        return performance_gflops > other.performance_gflops; // Higher is better
    }
};

class TuningHarness {
private:
    static double benchmark_kernel(std::function<void(const TuningParams&, int, int, int, float*, float*, float*)> kernel,
                                 const TuningParams& params, int M, int N, int K, int iterations = 10) {
        std::vector<float> A(M * K, 1.0f);
        std::vector<float> B(K * N, 1.0f); 
        std::vector<float> C(M * N, 0.0f);
        
        // Warm-up
        for (int i = 0; i < 3; i++) {
            kernel(params, M, N, K, A.data(), B.data(), C.data());
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            kernel(params, M, N, K, A.data(), B.data(), C.data());
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        double seconds = duration * 1e-9 / iterations;
        double operations = 2.0 * M * N * K; // FMA operations
        return operations / seconds * 1e-9; // GFLOPS
    }
    
public:
    template<typename ParamGenerator>
    static TuningResult tune_kernel(std::function<void(const TuningParams&, int, int, int, float*, float*, float*)> kernel,
                                   ParamGenerator param_generator,
                                   int M = 128, int N = 128, int K = 128) {
        std::vector<TuningResult> results;
        
        for (const auto& params : param_generator()) {
            double perf = benchmark_kernel(kernel, params, M, N, K);
            results.push_back({params, perf});
            
            std::cout << "Params: ";
            for (const auto& [name, value] : params.params) {
                std::cout << name << "=" << value << " ";
            }
            std::cout << "-> " << perf << " GFLOPS" << std::endl;
        }
        
        std::sort(results.begin(), results.end());
        
        std::cout << "\nBest configuration:" << std::endl;
        std::cout << "Performance: " << results[0].performance_gflops << " GFLOPS" << std::endl;
        std::cout << "Parameters: ";
        for (const auto& [name, value] : results[0].params.params) {
            std::cout << name << "=" << value << " ";
        }
        std::cout << std::endl;
        
        return results[0];
    }
    
    // Simple grid search helper
    static std::vector<TuningParams> grid_search(const std::map<std::string, std::vector<int>>& param_space) {
        std::vector<TuningParams> results;
        
        std::function<void(TuningParams, std::map<std::string, std::vector<int>>::const_iterator)> 
        generate = [&](TuningParams current, auto it) {
            if (it == param_space.end()) {
                results.push_back(current);
                return;
            }
            
            for (int value : it->second) {
                generate(TuningParams(current).set(it->first, value), std::next(it));
            }
        };
        
        generate(TuningParams{}, param_space.begin());
        return results;
    }
};