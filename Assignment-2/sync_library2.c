#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#include "barriers.h"

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