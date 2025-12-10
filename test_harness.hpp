#pragma once
#include <vector>
#include <random>
#include <cmath>
#include <iostream>

class TestHarness {
public:
    static void naive_matmul(int M, int N, int K, const float* A, const float* B, float* C) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }
    
    static bool test_kernel(void (*kernel)(int, int, int, float*, float*, float*), 
                           int M = 32, int N = 32, int K = 32, float tolerance = 1e-5f) {
        std::vector<float> A(M * K);
        std::vector<float> B(K * N);
        std::vector<float> C_expected(M * N);
        std::vector<float> C_actual(M * N);
        
        std::random_device rd;
        std::mt19937 gen(42); // Fixed seed for reproducibility
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        for (auto& val : A) val = dis(gen);
        for (auto& val : B) val = dis(gen);
        
        naive_matmul(M, N, K, A.data(), B.data(), C_expected.data());
        kernel(M, N, K, A.data(), B.data(), C_actual.data());
        
        for (int i = 0; i < M * N; i++) {
            if (std::abs(C_expected[i] - C_actual[i]) > tolerance) {
                std::cout << "Mismatch at index " << i << ": expected " 
                         << C_expected[i] << ", got " << C_actual[i] << std::endl;
                return false;
            }
        }
        
        std::cout << "Test passed for " << M << "x" << N << "x" << K << " matrix multiply" << std::endl;
        return true;
    }
    
    static bool comprehensive_test(void (*kernel)(int, int, int, float*, float*, float*)) {
        std::vector<std::tuple<int, int, int>> test_cases = {
            {1, 1, 1}, {4, 4, 4}, {16, 16, 16}, {32, 32, 32}, 
            {64, 32, 16}, {100, 50, 75}, {128, 128, 128}
        };
        
        for (const auto& [M, N, K] : test_cases) {
            if (!test_kernel(kernel, M, N, K)) {
                std::cout << "Test failed for " << M << "x" << N << "x" << K << std::endl;
                return false;
            }
        }
        
        std::cout << "All comprehensive tests passed!" << std::endl;
        return true;
    }
};