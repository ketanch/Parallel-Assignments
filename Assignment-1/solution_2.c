#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>

//#define LOCAL_TESTING

void InitializeInput(int n, float L[n][n], float *Y, float *X) {
    srand(time(NULL));

    float a = 10.0;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i+1; j++) {
            L[i][j] = (((float)rand()/(float)RAND_MAX) * a)+4;
            L[i][j] = (int)L[i][j];
        }
        for (int j = i+1; j < n; j++)
            L[i][j] = 0;
    }

    for (int i = 0; i < n; i++) {
        X[i] = (int)(((float)rand()/(float)RAND_MAX) * a)+2;
    }

    float sum;

    for (int i = 0; i < n; i++) {
        sum = 0;
        for (int j = 0; j < n; j++)
            sum += L[i][j] * X[j];
        Y[i] = sum;
    }
}

void InitializeFromFile(FILE *filep, int n, float L[n][n], float *Y) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < i+1; j++)
            fscanf(filep, "%f", &L[i][j]);

    for (int i = 0; i < n; i++)
        fscanf(filep, "%f", &Y[i]);

}

int main(int argc, char *argv[]) {
    
    int n;
    struct timeval tv0, tv1;
	struct timezone tz0, tz1;
    int nthreads;

#ifdef LOCAL_TESTING
    n = atoi(argv[2]);
    float L[n][n];
    float Y[n];
    float temp_X[n];
    InitializeInput(n, L, Y, temp_X);
    nthreads = atoi(argv[1]);
#else
    if (argc < 4) {
        printf("Use ./%s <inp file> <out file> <num threads>\n", __FILE__);
        exit(0);
    }
    FILE *inp_filep = fopen(argv[1], "r");
    FILE *out_filep = fopen(argv[2], "w");

    fscanf(inp_filep, "%d", &n);
    float L[n][n];
    float Y[n];
    InitializeFromFile(inp_filep, n, L, Y);
#endif
    float X[n];

    gettimeofday(&tv0, &tz0);

    for (int i = 0; i < n; i++) {
        X[i] = Y[i] / L[i][i];
#pragma omp parallel for num_threads(nthreads) schedule(static)
        for (int j = i; j < n; j++)
            Y[j] -= X[i] * L[j][i];
    }

    gettimeofday(&tv1, &tz1);

#ifdef LOCAL_TESTING

    float variance = 0;

    for (int i = 0; i < n; i++) {
        variance += (temp_X[i] - X[i]) * (temp_X[i] - X[i]);
        //printf("%d - %f %f %f %f\n", i, temp_X[i] - X[i], temp_X[i], X[i], variance);
    }
    printf("Variance = %f\n", variance);

#else

    for (int i = 0; i < n; i++) {
        fprintf(out_filep, "%f ", X[i]);
    }

#endif

    printf("time: %ld microseconds\n", (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec));

    return 0;
}