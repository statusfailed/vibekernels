#include <cblas.h>

void singlethread_reference(int M, int N, int K, float* A, float* B, float* C) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
}