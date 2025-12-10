#include <algorithm>
#include <immintrin.h>
#include <vector>

// Blocking parameters – tune empirically
static int MC = 128;
static int NC = 256;
static int KC = 128;

// 4x8 AVX2 micro-kernel: C[4 x 8] += A[4 x kc] * Bp[kc x 8]
static inline void microkernel_4x8(
    int kc,
    const float* A, int lda,      // lda = K
    const float* Bp,              // packed B panel
    float* C, int ldc)            // ldc = N
{
    __m256 c0 = _mm256_loadu_ps(C + 0 * ldc);
    __m256 c1 = _mm256_loadu_ps(C + 1 * ldc);
    __m256 c2 = _mm256_loadu_ps(C + 2 * ldc);
    __m256 c3 = _mm256_loadu_ps(C + 3 * ldc);

    const float* a0 = A + 0 * lda;
    const float* a1 = A + 1 * lda;
    const float* a2 = A + 2 * lda;
    const float* a3 = A + 3 * lda;

    const float* b = Bp;

    for (int k = 0; k < kc; ++k) {
        __m256 b_vec = _mm256_loadu_ps(b);
        b += 8;

        __m256 a0_b = _mm256_broadcast_ss(a0++);
        __m256 a1_b = _mm256_broadcast_ss(a1++);
        __m256 a2_b = _mm256_broadcast_ss(a2++);
        __m256 a3_b = _mm256_broadcast_ss(a3++);

        c0 = _mm256_fmadd_ps(a0_b, b_vec, c0);
        c1 = _mm256_fmadd_ps(a1_b, b_vec, c1);
        c2 = _mm256_fmadd_ps(a2_b, b_vec, c2);
        c3 = _mm256_fmadd_ps(a3_b, b_vec, c3);
    }

    _mm256_storeu_ps(C + 0 * ldc, c0);
    _mm256_storeu_ps(C + 1 * ldc, c1);
    _mm256_storeu_ps(C + 2 * ldc, c2);
    _mm256_storeu_ps(C + 3 * ldc, c3);
}

// Pack B block: Bp[ jblock ][ k ][ 8 columns ]
// Bp layout: for each 8-column block within nc:
//   for k in 0..kc-1: 8 contiguous floats (one row of that 8-wide sub-panel)
static void pack_B_panel(
    float* Bp,
    const float* B, int N,
    int kk, int jj,
    int kc, int nc_full)          // nc_full is multiple of 8
{
    float* dst = Bp;
    for (int jrel = 0; jrel < nc_full; jrel += 8) {
        int j = jj + jrel;
        for (int krel = 0; krel < kc; ++krel) {
            int k = kk + krel;
            const float* src_row = B + k * N + j;
            // copy 8 contiguous columns
            _mm256_storeu_ps(dst, _mm256_loadu_ps(src_row));
            dst += 8;
        }
    }
}

void singlethread_blocked_avx2_microkernel4x4_setup() {}
void singlethread_blocked_avx2_microkernel4x4_teardown() {}

void singlethread_blocked_avx2_microkernel4x4(int M, int N, int K, float* A, float* B, float* C) {
    // High-level structure: C += A * B
    // Outer blocking over N (NC) and K (KC), inner micro-kernel over M,N tiles.

    for (int jj = 0; jj < N; jj += NC) {
        int j_end = std::min(jj + NC, N);
        int nc = j_end - jj;
        int nc_full = (nc / 8) * 8; // columns handled by 4x8 kernel

        for (int kk = 0; kk < K; kk += KC) {
            int k_end = std::min(kk + KC, K);
            int kc = k_end - kk;

            // Pack B panel [kc x nc_full]
            std::vector<float> Bp;
            if (nc_full > 0) {
                Bp.resize(kc * nc_full);
                pack_B_panel(Bp.data(), B, N, kk, jj, kc, nc_full);
            }

            for (int ii = 0; ii < M; ii += MC) {
                int i_end = std::min(ii + MC, M);
                int mc = i_end - ii;
                int mc_full = (mc / 4) * 4; // rows handled by 4x8 kernel

                // Micro-kernel region: i in [ii, ii+mc_full), j in [jj, jj+nc_full)
                for (int i = ii; i < ii + mc_full; i += 4) {
                    for (int jrel = 0; jrel < nc_full; jrel += 8) {
                        int j = jj + jrel;

                        const float* A_block = A + i * K + kk;   // 4 x kc (row-major)
                        float* C_block = C + i * N + j;         // 4 x 8, ldc = N

                        // offset into packed B for this 8-wide block
                        int blk_index = jrel / 8;
                        const float* Bp_block = Bp.data() + blk_index * kc * 8;

                        microkernel_4x8(kc, A_block, K, Bp_block, C_block, N);
                    }
                }

                // Handle remaining rows (if mc not multiple of 4) using simple blocked AVX2/ scalar
                for (int i = ii + mc_full; i < i_end; ++i) {
                    // full 8-wide chunks within nc_full
                    int j = jj;
                    for (; j < jj + nc_full; j += 8) {
                        __m256 c_vec = _mm256_loadu_ps(&C[i * N + j]);
                        for (int k = kk; k < k_end; ++k) {
                            __m256 a_vec = _mm256_broadcast_ss(&A[i * K + k]);
                            __m256 b_vec = _mm256_loadu_ps(&B[k * N + j]);
                            c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                        }
                        _mm256_storeu_ps(&C[i * N + j], c_vec);
                    }
                    // tail columns (inside this jj..j_end block)
                    for (; j < j_end; ++j) {
                        float sum = C[i * N + j];
                        for (int k = kk; k < k_end; ++k) {
                            sum += A[i * K + k] * B[k * N + j];
                        }
                        C[i * N + j] = sum;
                    }
                }

                // Handle remaining columns (tail in N) for rows processed by micro-kernel
                for (int i = ii; i < ii + mc_full; ++i) {
                    for (int j = jj + nc_full; j < j_end; ++j) {
                        float sum = C[i * N + j];
                        for (int k = kk; k < k_end; ++k) {
                            sum += A[i * K + k] * B[k * N + j];
                        }
                        C[i * N + j] = sum;
                    }
                }
            }
        }
    }
}

