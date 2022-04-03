#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define N 1000000

int main(int argc, char **argv) {
    struct timeval tv0, tv1;
    struct timezone tz0, tz1;
    int i;
    int P = atoi(argv[1]);

    gettimeofday(&tv0, &tz0);

#pragma omp parallel num_threads(P) private(i)
    {
        for (i = 0; i < N; i++) {
#pragma omp barrier
        }
    }

    gettimeofday(&tv1, &tz1);

    printf("Time: %lf seconds\n", ((tv1.tv_sec - tv0.tv_sec) * 1000000 + (tv1.tv_usec - tv0.tv_usec)) / 1000000.0);

    return 0;
}