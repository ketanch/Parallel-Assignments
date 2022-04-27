#include <assert.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

int P, n;
float **A, *B, *C, diff = 0.0;

void Initialize(float **X, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) X[i][j] = ((float)(random() % 100) / 100.0);
}

void Initialize1(float *X, int n) {
    for (int i = 0; i < n; i++)
        X[i] = ((float)(random() % 100) / 100.0);
}

int main(int argc, char **argv) {
    struct timeval tv0, tv1;
    struct timezone tz0, tz1;
    char buffer[64];

    if (argc != 3) {
        printf("Need grid size (n) and number of threads (P).\nAborting...\n");
        exit(1);
    }

    n = atoi(argv[1]);
    P = atoi(argv[2]);

    A = (float **)malloc((n) * sizeof(float *));
    assert(A != NULL);
    for (int i = 0; i < n; i++) {
        A[i] = (float *)malloc((n) * sizeof(float));
        assert(A[i] != NULL);
    }
    B = (float *)malloc(n * sizeof(float));
    C = (float *)calloc(n, sizeof(float));
    Initialize(A, n);
    Initialize1(B, n);

    gettimeofday(&tv0, &tz0);
#pragma omp parallel for num_threads(P)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i] += A[i][j] * B[j];
        }
    }
    gettimeofday(&tv1, &tz1);
    printf("Time: %f seconds\n", ((tv1.tv_sec - tv0.tv_sec) * 1000000 + (tv1.tv_usec - tv0.tv_usec)) / 1000000.0);
}