#include <cblas.h>
#include <omp.h>

void singlethread_reference_setup() {
    omp_set_num_threads(1);
    openblas_set_num_threads(1);
}

void singlethread_reference_teardown() {
    // Could restore original values if needed
}

void singlethread_reference(int M, int N, int K, float* A, float* B, float* C) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
}
