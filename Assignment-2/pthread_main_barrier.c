#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "barriers.h"

#define N 1000000

int P;
struct Central_Sense_Reversing_t barrier;

void *forloop(void *p) {
    int localsense = 1;
    for (int i = 0; i < N; i++) {
        Central_Sense_Reversing_Wait(&barrier, &localsense, P);
    }
    return NULL;
}

int main(int argc, char **argv) {
    struct timeval tv0, tv1;
    struct timezone tz0, tz1;
    P = atoi(argv[1]);

    gettimeofday(&tv0, &tz0);

    pthread_attr_t attr;
    pthread_t *tid;

    Central_Sense_Reversing_Init(&barrier);
    tid = (pthread_t *)malloc(P * sizeof(pthread_t));
    pthread_attr_init(&attr);

    for (int i = 0; i < P; i++) {
        pthread_create(&tid[i], &attr, forloop, NULL);
    }

    for (int i = 0; i < P; i++) {
        pthread_join(tid[i], NULL);
    }

    gettimeofday(&tv1, &tz1);

    printf("Time: %lf seconds\n", ((tv1.tv_sec - tv0.tv_sec) * 1000000 + (tv1.tv_usec - tv0.tv_usec)) / 1000000.0);

    return 0;
}