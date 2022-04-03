#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define CACHE_LINE_SIZE 64

extern void Init_Lamport_Bakery(int nthread, int **choosing, int **ticket);
extern void Acquire_Lamport_Bakery(int pid, int nthreads, int *choosing, int *ticket);
extern void Release_Lamport_Bakery(int pid, int *choosing, int *ticket);

extern void Acquire_Spinlock(int *lock_addr);
extern void Release_Spinlock(int *lock_addr);

extern void Acquire_TTS(int *addr);
extern void Release_TTS(int *addr);

extern void Acquire_Ticket_Lock(int *ticket_addr, int *release_count_addr);
extern void Release_Ticket_Lock(int *release_count_addr);

extern char* Init_Array_Lock(int len);
extern int Acquire_Array_Lock(char *lock_arr, int arr_len, int *index);
extern void Release_Array_Lock(char *lock_addr, int th_index, int);

extern void Acquire_pthread_mutex (pthread_mutex_t lock);
extern void Release_pthread_mutex (pthread_mutex_t lock);