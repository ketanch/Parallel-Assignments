#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#include "barriers.h"

const int CACHE_LINE_SIZE = 64;

void Central_Sense_Reversing_Init(struct Central_Sense_Reversing_t *barrier) {
    barrier->counter = 0;
    barrier->flag = 1;
    pthread_mutex_init(&barrier->mutex, NULL);
}

void Central_Sense_Reversing_Wait(struct Central_Sense_Reversing_t *barrier, int *localsense, int num_threads) {
    // here localsense must be private to each thread initialized to 1
    *localsense = 1 - *localsense;
    pthread_mutex_lock(&barrier->mutex);
    barrier->counter++;
    if (barrier->counter == num_threads) {
        pthread_mutex_unlock(&barrier->mutex);
        barrier->counter = 0;
        barrier->flag = *localsense;
    } else {
        pthread_mutex_unlock(&barrier->mutex);
        while (barrier->flag != *localsense) {
            asm("" ::
                    : "memory");
        }
    }
}

void Central_Posix_CV_Init(Central_Posix_CV_t *barrier) {
    barrier->counter = 0;
    pthread_cond_init(&barrier->cv, NULL);
    pthread_mutex_init(&barrier->mutex, NULL);
}

void Central_Posix_CV_Wait(Central_Posix_CV_t *barrier, int num_threads) {
    pthread_mutex_lock(&barrier->mutex);
    barrier->counter++;
    if (barrier->counter == num_threads) {
        barrier->counter = 0;
        pthread_cond_broadcast(&barrier->cv);
    } else {
        pthread_cond_wait(&barrier->cv, &barrier->mutex);
    }
    pthread_mutex_unlock(&barrier->mutex);
}

void Tree_Sense_Reversing_Init(Tree_Sense_Reversing_t *barrier, int num_threads) {
    barrier->flag = (int **)malloc(sizeof(int *) * num_threads);
    for (int i = 0; i < num_threads; i++) {
        barrier->flag[i] = (int *)malloc(sizeof(int) * num_threads * CACHE_LINE_SIZE);

        for (int j = 0; j < num_threads; j++) {
            barrier->flag[i][j*CACHE_LINE_SIZE] = 0;
        }
    }
}

void Tree_Sense_Reversing_Wait(Tree_Sense_Reversing_t *barrier, int thread_id, int num_threads) {
    unsigned int i, mask;
    for (i = 0, mask = 1; (mask & thread_id) != 0; i++, mask <<= 1) {
        while (!barrier->flag[thread_id][i*CACHE_LINE_SIZE]) {
            asm("" ::
                    : "memory");
        }
        barrier->flag[thread_id][i*CACHE_LINE_SIZE] = 0;
    }

    if ((thread_id < num_threads - 1) && (thread_id + mask <= num_threads - 1)) {
        barrier->flag[thread_id + mask][i*CACHE_LINE_SIZE] = 1;
        while (!barrier->flag[thread_id][(num_threads - 1)*CACHE_LINE_SIZE]) {
            asm("" ::
                    : "memory");
        }
        barrier->flag[thread_id][(num_threads - 1)*CACHE_LINE_SIZE] = 0;
    }
    for (mask >>= 1; mask != 0; mask >>= 1) {
        barrier->flag[thread_id - mask][(num_threads - 1)*CACHE_LINE_SIZE] = 1;
    }
}

// }
//
// void Tree_Posix_CV_Init(Tree_Posix_CV_t *barrier, int num_threads) {
//     barrier->cv = (pthread_cond_t **)malloc(sizeof(pthread_cond_t *) * num_threads);
//     for (int i = 0; i < num_threads; i++) {
//         barrier->cv[i] = (pthread_cond_t *)malloc(sizeof(pthread_cond_t) * num_threads);
//         for (int j = 0; j < num_threads; j++) {
//             pthread_cond_init(&barrier->cv[i][j], NULL);
//         }
//     }
//     barrier->mutex = (pthread_mutex_t *)malloc(sizeof(pthread_mutex_t) * num_threads);
//     for (int i = 0; i < num_threads; i++) {
//         pthread_mutex_init(&barrier->mutex[i], NULL);

// void Tree_Posix_CV_Wait(Tree_Posix_CV_t *barrier, int thread_id, int num_threads) {
//     unsigned int i, mask;
//     for (i = 0, mask = 1; (mask & thread_id) != 0; i++, mask <<= 1) {
//         pthread_mutex_lock(&barrier->mutex[thread_id]);
//         pthread_cond_wait(&barrier->cv[thread_id][i], &barrier->mutex[thread_id]);
//         printf("un %d\n", thread_id);
//         pthread_mutex_unlock(&barrier->mutex[thread_id]);
//         pthread_cond_broadcast(&barrier->cv[thread_id][i]);
//     }

//     if ((thread_id < num_threads - 1) && (thread_id + mask <= num_threads - 1)) {
//         // printf("%d\n", thread_id);

//         printf("broadcasted i: %d, j: %d\n", thread_id + mask, i);
//         sleep(2);
//         pthread_cond_broadcast(&barrier->cv[thread_id + mask][i]);
//         pthread_mutex_lock(&barrier->mutex[thread_id]);
//         pthread_cond_wait(&barrier->cv[thread_id][num_threads - 1], &barrier->mutex[thread_id]);
//         pthread_mutex_unlock(&barrier->mutex[thread_id]);
//         pthread_cond_broadcast(&barrier->cv[thread_id][num_threads - 1]);
//     }
//     for (mask >>= 1; mask != 0; mask >>= 1) {
//         // printf("asa %d\n", thread_id);

//         pthread_cond_broadcast(&barrier->cv[thread_id - mask][num_threads - 1]);
//     }
// }
