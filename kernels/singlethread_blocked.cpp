#include <algorithm>

// Cache blocking parameters - will be tuned
static int BLOCK_SIZE = 64;

void singlethread_blocked_setup() {
    // Setup function for any initialization
}

void singlethread_blocked_teardown() {
    // Cleanup if needed
}

void singlethread_blocked(int M, int N, int K, float* A, float* B, float* C) {
    // Cache-blocked (tiled) matrix multiplication
    // Divide computation into blocks to improve cache locality
    
    for (int ii = 0; ii < M; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
            for (int kk = 0; kk < K; kk += BLOCK_SIZE) {
                
                // Determine block boundaries
                int i_end = std::min(ii + BLOCK_SIZE, M);
                int j_end = std::min(jj + BLOCK_SIZE, N);
                int k_end = std::min(kk + BLOCK_SIZE, K);
                
                // Compute within the block
                for (int i = ii; i < i_end; i++) {
                    for (int j = jj; j < j_end; j++) {
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