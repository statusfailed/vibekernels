#include <cblas.h>
#include <omp.h>

void multithread_reference_setup() {
    openblas_set_num_threads(omp_get_max_threads());
}

void multithread_reference_teardown() {}

void multithread_reference(int M, int N, int K, float* A, float* B, float* C) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
}
