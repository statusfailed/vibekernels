#include <cblas.h>

void multithread_reference_setup() {}

void multithread_reference_teardown() {}

void multithread_reference(int M, int N, int K, float* A, float* B, float* C) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
}
