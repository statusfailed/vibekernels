#include <algorithm>
#include <immintrin.h>
#include <vector>

// Blocking parameters – tune empirically for AVX512
static int MC = 128;
static int NC = 256;
static int KC = 128;

// 4x16 AVX512 micro-kernel: C[4 x 16] += A[4 x kc] * Bp[kc x 16]
static inline void microkernel_4x16(
    int kc,
    const float* A, int lda,      // lda = K
    const float* Bp,              // packed B panel
    float* C, int ldc)            // ldc = N
{
    __m512 c0 = _mm512_loadu_ps(C + 0 * ldc);
    __m512 c1 = _mm512_loadu_ps(C + 1 * ldc);
    __m512 c2 = _mm512_loadu_ps(C + 2 * ldc);
    __m512 c3 = _mm512_loadu_ps(C + 3 * ldc);

    const float* a0 = A + 0 * lda;
    const float* a1 = A + 1 * lda;
    const float* a2 = A + 2 * lda;
    const float* a3 = A + 3 * lda;

    const float* b = Bp;

    for (int k = 0; k < kc; ++k) {
        __m512 b_vec = _mm512_loadu_ps(b);
        b += 16;

        __m512 a0_b = _mm512_set1_ps(*a0++);
        __m512 a1_b = _mm512_set1_ps(*a1++);
        __m512 a2_b = _mm512_set1_ps(*a2++);
        __m512 a3_b = _mm512_set1_ps(*a3++);

        c0 = _mm512_fmadd_ps(a0_b, b_vec, c0);
        c1 = _mm512_fmadd_ps(a1_b, b_vec, c1);
        c2 = _mm512_fmadd_ps(a2_b, b_vec, c2);
        c3 = _mm512_fmadd_ps(a3_b, b_vec, c3);
    }

    _mm512_storeu_ps(C + 0 * ldc, c0);
    _mm512_storeu_ps(C + 1 * ldc, c1);
    _mm512_storeu_ps(C + 2 * ldc, c2);
    _mm512_storeu_ps(C + 3 * ldc, c3);
}

// Pack B block: Bp[ jblock ][ k ][ 16 columns ]
// Bp layout: for each 16-column block within nc:
//   for k in 0..kc-1: 16 contiguous floats (one row of that 16-wide sub-panel)
static void pack_B_panel(
    float* Bp,
    const float* B, int N,
    int kk, int jj,
    int kc, int nc_full)          // nc_full is multiple of 16
{
    float* dst = Bp;
    for (int jrel = 0; jrel < nc_full; jrel += 16) {
        int j = jj + jrel;
        for (int krel = 0; krel < kc; ++krel) {
            int k = kk + krel;
            const float* src_row = B + k * N + j;
            // copy 16 contiguous columns
            _mm512_storeu_ps(dst, _mm512_loadu_ps(src_row));
            dst += 16;
        }
    }
}

void singlethread_blocked_avx512_microkernel4x16_setup() {}
void singlethread_blocked_avx512_microkernel4x16_teardown() {}

void singlethread_blocked_avx512_microkernel4x16(int M, int N, int K, float* A, float* B, float* C) {
    // High-level structure: C += A * B
    // Outer blocking over N (NC) and K (KC), inner micro-kernel over M,N tiles.

    for (int jj = 0; jj < N; jj += NC) {
        int j_end = std::min(jj + NC, N);
        int nc = j_end - jj;
        int nc_full = (nc / 16) * 16; // columns handled by 4x16 kernel

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
                int mc_full = (mc / 4) * 4; // rows handled by 4x16 kernel

                // Micro-kernel region: i in [ii, ii+mc_full), j in [jj, jj+nc_full)
                for (int i = ii; i < ii + mc_full; i += 4) {
                    for (int jrel = 0; jrel < nc_full; jrel += 16) {
                        int j = jj + jrel;

                        const float* A_block = A + i * K + kk;   // 4 x kc (row-major)
                        float* C_block = C + i * N + j;         // 4 x 16, ldc = N

                        // offset into packed B for this 16-wide block
                        int blk_index = jrel / 16;
                        const float* Bp_block = Bp.data() + blk_index * kc * 16;

                        microkernel_4x16(kc, A_block, K, Bp_block, C_block, N);
                    }
                }

                // Handle remaining rows (if mc not multiple of 4) using simple blocked AVX512/scalar
                for (int i = ii + mc_full; i < i_end; ++i) {
                    // full 16-wide chunks within nc_full
                    int j = jj;
                    for (; j < jj + nc_full; j += 16) {
                        __m512 c_vec = _mm512_loadu_ps(&C[i * N + j]);
                        for (int k = kk; k < k_end; ++k) {
                            __m512 a_vec = _mm512_set1_ps(A[i * K + k]);
                            __m512 b_vec = _mm512_loadu_ps(&B[k * N + j]);
                            c_vec = _mm512_fmadd_ps(a_vec, b_vec, c_vec);
                        }
                        _mm512_storeu_ps(&C[i * N + j], c_vec);
                    }
                    // 8-wide chunks (fallback to AVX2 for better coverage)
                    for (; j < jj + nc_full && j + 8 <= j_end; j += 8) {
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