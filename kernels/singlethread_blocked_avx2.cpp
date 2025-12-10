#include <algorithm>
#include <immintrin.h>

// Cache blocking parameters - will be tuned
static int BLOCK_SIZE = 64;

void singlethread_blocked_avx2_setup() {
    // Setup function for any initialization
}

void singlethread_blocked_avx2_teardown() {
    // Cleanup if needed
}

void singlethread_blocked_avx2(int M, int N, int K, float* A, float* B, float* C) {
    // Cache-blocked matrix multiplication with AVX2 vectorization
    // AVX2 provides 8-way parallelism for single-precision floats
    
    for (int ii = 0; ii < M; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
            for (int kk = 0; kk < K; kk += BLOCK_SIZE) {
                
                // Determine block boundaries
                int i_end = std::min(ii + BLOCK_SIZE, M);
                int j_end = std::min(jj + BLOCK_SIZE, N);
                int k_end = std::min(kk + BLOCK_SIZE, K);
                
                // Compute within the block using AVX2
                for (int i = ii; i < i_end; i++) {
                    // Process 8 elements at a time using AVX2
                    int j = jj;
                    for (; j + 8 <= j_end; j += 8) {
                        // Load C[i][j:j+8]
                        __m256 c_vec = _mm256_loadu_ps(&C[i * N + j]);
                        
                        // Accumulate over k
                        for (int k = kk; k < k_end; k++) {
                            // Broadcast A[i][k] to all 8 lanes
                            __m256 a_vec = _mm256_broadcast_ss(&A[i * K + k]);
                            
                            // Load B[k][j:j+8]
                            __m256 b_vec = _mm256_loadu_ps(&B[k * N + j]);
                            
                            // Fused multiply-add: c_vec += a_vec * b_vec
                            c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                        }
                        
                        // Store result back to C[i][j:j+8]
                        _mm256_storeu_ps(&C[i * N + j], c_vec);
                    }
                    
                    // Handle remaining elements (if N is not multiple of 8)
                    for (; j < j_end; j++) {
                        float sum = C[i * N + j];
                        for (int k = kk; k < k_end; k++) {
                            sum += A[i * K + k] * B[k * N + j];
                        }
                        C[i * N + j] = sum;
                    }
                }
            }
        }
    }
}