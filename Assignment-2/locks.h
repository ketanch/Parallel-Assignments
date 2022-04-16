#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <semaphore.h>

#define CACHE_LINE_SIZE 64
#define INT_COUNT 16

void Init_Lamport_Bakery(int nthread, int **choosing, int **ticket);
void Acquire_Lamport_Bakery(int pid, int nthreads, int *choosing, int *ticket);
void Release_Lamport_Bakery(int pid, int *choosing, int *ticket);

void Acquire_Spinlock(int *lock_addr);
void Release_Spinlock(int *lock_addr);

void Acquire_TTS(int *addr);
void Release_TTS(int *addr);

void Acquire_Ticket_Lock(int *ticket_addr, int *release_count_addr);
void Release_Ticket_Lock(int *release_count_addr);

char *Init_Array_Lock(int len);
int Acquire_Array_Lock(char *lock_arr, int arr_len, int *index);
void Release_Array_Lock(char *lock_addr, int th_index, int);

void Acquire_pthread_mutex(pthread_mutex_t *lock);
void Release_pthread_mutex(pthread_mutex_t *lock);

void Acquire_sema_lock(sem_t *sema_lock);
void Release_sema_lock(sem_t *sema_lock);