#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

//#define LOCAL_TESTING

void InitializeInput(int n, float L[n][n], float *Y) {
    srand(time(NULL));
    float a = 20.0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < i+1; j++) {
            L[i][j] = (((float)rand()/(float)RAND_MAX) * a) - a/2;
            if (L[i][j] == 0)
                printf("--------------------\n");
        }

    for (int i = 0; i < n; i++)
        Y[i] = (((float)rand()/(float)RAND_MAX) * a) - a/2;
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

#ifdef LOCAL_TESTING
    n = 100;
    float L[n][n];
    float Y[n];
    InitializeInput(n, L, Y);
#else
    if (argc < 2) {
        printf("Use ./%s <inp file> <out file>\n", __FILE__);
        exit(0);
    }
    FILE *filep = fopen(argv[1], "r");
    fscanf(filep, "%d", &n);
    float L[n][n];
    float Y[n];
    InitializeFromFile(filep, n, L, Y);
#endif
    float X[n];

    for (int i = 0; i < n; i++) {
        X[i] = Y[i] / L[i][i];
        for (int j = i; j < n; j++)
            Y[j] -= X[i] * L[j][i];
    }

    for (int i = 0; i < n; i++) {
        printf("%d %f\n", i, X[i]);
    }
    printf("\n");

    return 0;
}