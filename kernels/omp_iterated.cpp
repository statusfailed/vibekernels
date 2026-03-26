#include <algorithm>
#include <immintrin.h>
#include <omp.h>
#include <vector>

// Threaded decomposition over N panels. Each thread owns a column panel of C,
// packs the matching panel of B, and runs the in-repo AVX-512 microkernel.
static constexpr int PANEL_N = 256;
static constexpr int KC = 256;

static constexpr int PREFETCH_DISTANCE_A = 8;
static constexpr int PREFETCH_DISTANCE_B = 4;

static inline void microkernel_4x16_prefetch(
    int kc,
    const float* A, int lda,
    const float* Bp,
    float* C, int ldc,
    const float* A_next = nullptr)
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
        if (k + PREFETCH_DISTANCE_B < kc) {
            _mm_prefetch(reinterpret_cast<const char*>(b + PREFETCH_DISTANCE_B * 16), _MM_HINT_T0);
        }

        if (A_next && k < kc - 1) {
            _mm_prefetch(reinterpret_cast<const char*>(A_next + k * lda), _MM_HINT_T0);
        } else if (k + PREFETCH_DISTANCE_A < kc) {
            _mm_prefetch(reinterpret_cast<const char*>(a0 + PREFETCH_DISTANCE_A), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(a1 + PREFETCH_DISTANCE_A), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(a2 + PREFETCH_DISTANCE_A), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(a3 + PREFETCH_DISTANCE_A), _MM_HINT_T0);
        }

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

static void pack_B_panel_prefetch(
    float* Bp,
    const float* B, int N,
    int kk, int jj,
    int kc, int nc_full)
{
    float* dst = Bp;
    for (int jrel = 0; jrel < nc_full; jrel += 16) {
        int j = jj + jrel;
        for (int krel = 0; krel < kc; ++krel) {
            int k = kk + krel;
            const float* src_row = B + k * N + j;

            if (krel + 2 < kc) {
                _mm_prefetch(reinterpret_cast<const char*>(B + (k + 2) * N + j), _MM_HINT_T0);
            }

            _mm512_storeu_ps(dst, _mm512_loadu_ps(src_row));
            dst += 16;
        }
    }
}

void omp_iterated_setup() {
    omp_set_dynamic(0);
}

void omp_iterated_teardown() {}

void omp_iterated(int M, int N, int K, float* A, float* B, float* C) {
    #pragma omp parallel for schedule(static)
    for (int jj = 0; jj < N; jj += PANEL_N) {
        int j_end = std::min(jj + PANEL_N, N);
        int nc = j_end - jj;
        int nc_full = (nc / 16) * 16;

        std::vector<float> Bp;

        for (int kk = 0; kk < K; kk += KC) {
            int k_end = std::min(kk + KC, K);
            int kc = k_end - kk;

            if (nc_full > 0) {
                Bp.resize(kc * nc_full);
                pack_B_panel_prefetch(Bp.data(), B, N, kk, jj, kc, nc_full);
            }

            int m_full = (M / 4) * 4;

            for (int i = 0; i < m_full; i += 4) {
                for (int jrel = 0; jrel < nc_full; jrel += 16) {
                    int j = jj + jrel;
                    const float* A_block = A + i * K + kk;
                    float* C_block = C + i * N + j;
                    int blk_index = jrel / 16;
                    const float* Bp_block = Bp.data() + blk_index * kc * 16;

                    const float* A_next = nullptr;
                    if (i + 4 < m_full) {
                        A_next = A + (i + 4) * K + kk;
                    }

                    microkernel_4x16_prefetch(kc, A_block, K, Bp_block, C_block, N, A_next);
                }
            }

            for (int i = m_full; i < M; ++i) {
                int j = jj;
                for (; j < jj + nc_full; j += 16) {
                    __m512 c_vec = _mm512_loadu_ps(&C[i * N + j]);
                    for (int k = kk; k < k_end; ++k) {
                        if (k + 2 < k_end) {
                            _mm_prefetch(reinterpret_cast<const char*>(&A[i * K + k + 2]), _MM_HINT_T0);
                            _mm_prefetch(reinterpret_cast<const char*>(&B[(k + 2) * N + j]), _MM_HINT_T0);
                        }

                        __m512 a_vec = _mm512_set1_ps(A[i * K + k]);
                        __m512 b_vec = _mm512_loadu_ps(&B[k * N + j]);
                        c_vec = _mm512_fmadd_ps(a_vec, b_vec, c_vec);
                    }
                    _mm512_storeu_ps(&C[i * N + j], c_vec);
                }

                for (; j < jj + nc_full && j + 8 <= j_end; j += 8) {
                    __m256 c_vec = _mm256_loadu_ps(&C[i * N + j]);
                    for (int k = kk; k < k_end; ++k) {
                        if (k + 2 < k_end) {
                            _mm_prefetch(reinterpret_cast<const char*>(&A[i * K + k + 2]), _MM_HINT_T0);
                            _mm_prefetch(reinterpret_cast<const char*>(&B[(k + 2) * N + j]), _MM_HINT_T0);
                        }

                        __m256 a_vec = _mm256_broadcast_ss(&A[i * K + k]);
                        __m256 b_vec = _mm256_loadu_ps(&B[k * N + j]);
                        c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                    }
                    _mm256_storeu_ps(&C[i * N + j], c_vec);
                }

                for (; j < j_end; ++j) {
                    float sum = C[i * N + j];
                    for (int k = kk; k < k_end; ++k) {
                        sum += A[i * K + k] * B[k * N + j];
                    }
                    C[i * N + j] = sum;
                }
            }

            for (int i = 0; i < m_full; ++i) {
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
