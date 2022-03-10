#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>

float** construct_L(int n) {
    float *ptr = (float*) malloc(((n*(n+1))/2) * sizeof(float));
    float **L = (float**) malloc(n * sizeof(float*));
    for (int i = 0; i < n; i++) {
        L[i] = ptr + (i*(i+1))/2;
    }
    return L;
}

void InitializeInput(int n, float **L, float *Y, float *X) {

    for(int i = 0; i<n; i++){
        for(int j = 0; j<i; j++){
            L[i][j] = (1.0+j+i)/(j*i + 4);
        }
        L[i][i] = (5.0*i)/(1.0*i + 8.0);
    }
    L[0][0] = 1;

    for(int i = 0; i<n; i++)
        X[i] = (i*2.0 + 5)/(i+10);
    
    float sum;
    for (int i = 0; i < n; i++) {
        sum = 0;
        for (int j = 0; j < i+1; j++)
            sum += L[i][j] * X[j];
        Y[i] = sum;
    }
}

void InitializeFromFile(FILE *filep, int n, float **L, float *Y) {
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
    float **L = construct_L(n);
    float *Y = (float*) malloc(n * sizeof(float));
    float *temp_X = (float*) malloc(n * sizeof(float));
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
    float **L = construct_L(n);
    float *Y = (float*) malloc(n * sizeof(float));
    InitializeFromFile(inp_filep, n, L, Y);
    fclose(inp_filep);

#endif

    float *X = (float*) malloc(n * sizeof(float));

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
    int i = 0;
    for (; i < n; i++) {
        variance += (temp_X[i] - X[i]) * (temp_X[i] - X[i]);
    }
    printf("Variance = %f\n", variance / i);
    free(temp_X);

#else

    for (int i = 0; i < n; i++) {
        fprintf(out_filep, "%f ", X[i]);
    }
    fclose(out_filep);

#endif

    printf("time: %ld microseconds\n", (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec));
    free((void*)L);
    free(X);
    free(Y);

    return 0;
}