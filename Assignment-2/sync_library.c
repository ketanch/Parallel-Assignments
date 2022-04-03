#include "barriers.h"
#include "locks.h"

/*Computes maximum value in an array 'arr' from 0 to len-1*/
int max(int *arr, int len) {
    int max = arr[0];
    for (int i = 1; i < len; i++) {
        if (max < arr[i * CACHE_LINE_SIZE])
            max = arr[i * CACHE_LINE_SIZE];
    }
    return max;
}

/*Implements Atomic compare and set operation*/
unsigned char CompareAndSet(int old_val, int new_val, int *addr) {
    int old_val_out;
    unsigned char result;

    asm(
        "lock cmpxchgl %4, %1 \n\t"
        "setzb %0"
        : "=qm"(result), "+m"(*addr), "=a"(old_val_out)
        : "a"(old_val), "r"(new_val)
        :);

    return result;
}

/*Implements test and set operation using xchg*/
int TestAndSet(int *addr) {
    int old_val = 1;

    asm(
        "lock xchgl %0, %1 \n\t"
        : "+r"(old_val), "+m"(*addr)::);

    return old_val;
}

/*Implements fetch and increment using cmpxchg*/
int FetchAndInc(int *addr) {
    int old_val, new_val, res;
GetLock:
    old_val = *addr;
    new_val = old_val + 1;
    res = CompareAndSet(old_val, new_val, addr);
    if (!res)
        goto GetLock;

    return old_val;
}

/* Acquire for POSIX mutex */
void Acquire_pthread_mutex(pthread_mutex_t *lock) {
    pthread_mutex_lock(lock);
}

/* Release for POSIX mutex */
void Release_pthread_mutex(pthread_mutex_t *lock) {
    pthread_mutex_unlock(lock);
}

/*Initializes arrays for Lamport Bakery*/
void Init_Lamport_Bakery(int nthread, int **choosing, int **ticket) {
    *ticket = (int *)malloc(nthread * CACHE_LINE_SIZE);
    for (int i = 0; i < nthread; i++) {
        (*ticket)[i * CACHE_LINE_SIZE] = 0;
    }
    *choosing = (int *)malloc(nthread * CACHE_LINE_SIZE);
    for (int i = 0; i < nthread; i++) {
        (*choosing)[i * CACHE_LINE_SIZE] = 0;
    }
}

/*Acquire for Lamport Bakery*/
void Acquire_Lamport_Bakery(int pid, int nthreads, int *choosing, int *ticket) {
    choosing[pid * CACHE_LINE_SIZE] = 1;
    asm("mfence" ::
            : "memory");
    ticket[pid * CACHE_LINE_SIZE] = max(ticket, nthreads) + 1;
    asm("mfence" ::
            : "memory");
    choosing[pid * CACHE_LINE_SIZE] = 0;
    asm("mfence" ::
            : "memory");
    for (int i = 0; i < nthreads; i++) {
        while (choosing[i * CACHE_LINE_SIZE]) {
            asm("" ::
                    : "memory");
        }
        int ticket_i = ticket[i * CACHE_LINE_SIZE];
        int ticket_pid = ticket[pid * CACHE_LINE_SIZE];
        while (ticket_i && (ticket_i < ticket_pid || (ticket_i == ticket_pid && i < pid))) {
            ticket_i = ticket[i * CACHE_LINE_SIZE];
            ticket_pid = ticket[pid * CACHE_LINE_SIZE];
            asm("" ::
                    : "memory");
        }
    }
}

/*Release for Lamport Bakery*/
void Release_Lamport_Bakery(int pid, int *choosing, int *ticket) {
    asm("" ::
            : "memory");
    ticket[pid * CACHE_LINE_SIZE] = 0;
}

/*Acquire for Spinlock*/
void Acquire_Spinlock(int *lock_addr) {
    while (!CompareAndSet(0, 1, lock_addr))
        ;
}

/*Release for Spinlock*/
void Release_Spinlock(int *lock_addr) {
    asm("" ::
            : "memory");
    *lock_addr = 0;
}

/*Acquire for Test & test & set Lock*/
void Acquire_TTS(int *addr) {
    int reg_val;
Lock:
    reg_val = TestAndSet(addr);
    if (reg_val == 0) return;
    while (*addr != 0) {
        asm("" ::
                : "memory");
    }
    goto Lock;
}

/*Release for Test & test & set lock*/
void Release_TTS(int *addr) {
    asm("" ::
            : "memory");
    *addr = 0;
}

/*Acquire for Ticket Lock*/
void Acquire_Ticket_Lock(int *ticket_addr, int *release_count_addr) {
    int ticket = FetchAndInc(ticket_addr);
    while (ticket != *release_count_addr) {
        asm("" ::
                : "memory");
    }
    return;
}

/*Release for Ticket lock*/
void Release_Ticket_Lock(int *release_count_addr) {
    asm("" ::
            : "memory");
    *release_count_addr += 1;
}

/*Initializes array for Array lock*/
char *Init_Array_Lock(int len) {
    char *arr = (char *)malloc(len * sizeof(char) * CACHE_LINE_SIZE);
    for (int i = 0; i < len; i++) {
        arr[i * CACHE_LINE_SIZE] = 0;
    }
    arr[0] = 1;
    return arr;
}

/*Acquire for Array Lock*/
int Acquire_Array_Lock(char *lock_arr, int arr_len, int *index) {
    int th_index = FetchAndInc(index);
    th_index %= arr_len;
    while (!lock_arr[th_index * CACHE_LINE_SIZE]) {
        asm("" ::
                : "memory");
    }
    return th_index;
}

/*Release for Array Lock*/
void Release_Array_Lock(char *lock_addr, int th_index, int arr_len) {
    asm("" ::
            : "memory");
    lock_addr[th_index * CACHE_LINE_SIZE] = 0;
    lock_addr[((th_index + 1) % arr_len) * CACHE_LINE_SIZE] = 1;
}

/* Initialize Centralized sense-reversing barrier using busy-wait on flag  */
void Central_Sense_Reversing_Init(Central_Sense_Reversing_t *barrier) {
    barrier->counter = 0;
    barrier->flag = 1;
    pthread_mutex_init(&barrier->mutex, NULL);
}

/* Centralized sense-reversing barrier using busy-wait on flag */
void Central_Sense_Reversing_Wait(Central_Sense_Reversing_t *barrier, int *localsense, int num_threads) {
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

/* Initialize Centralized barrier using POSIX condition variable  */
void Central_Posix_CV_Init(Central_Posix_CV_t *barrier) {
    barrier->counter = 0;
    pthread_cond_init(&barrier->cv, NULL);
    pthread_mutex_init(&barrier->mutex, NULL);
}

/* Centralized barrier using POSIX condition variable  */
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

/* Initialize Tree barrier using busy-wait on flags  */
void Tree_Sense_Reversing_Init(Tree_Sense_Reversing_t *barrier, int num_threads) {
    barrier->flag = (int **)malloc(sizeof(int *) * num_threads);
    for (int i = 0; i < num_threads; i++) {
        barrier->flag[i] = (int *)malloc(sizeof(int) * num_threads * CACHE_LINE_SIZE);

        for (int j = 0; j < num_threads; j++) {
            barrier->flag[i][j * CACHE_LINE_SIZE] = 0;
        }
    }
}

/* Tree barrier using busy-wait on flags  */
void Tree_Sense_Reversing_Wait(Tree_Sense_Reversing_t *barrier, int thread_id, int num_threads) {
    unsigned int i, mask;
    for (i = 0, mask = 1; (mask & thread_id) != 0; i++, mask <<= 1) {
        while (!barrier->flag[thread_id][i * CACHE_LINE_SIZE]) {
            asm("" ::
                    : "memory");
        }
        barrier->flag[thread_id][i * CACHE_LINE_SIZE] = 0;
    }

    if ((thread_id < num_threads - 1) && (thread_id + mask <= num_threads - 1)) {
        barrier->flag[thread_id + mask][i * CACHE_LINE_SIZE] = 1;
        while (!barrier->flag[thread_id][(num_threads - 1) * CACHE_LINE_SIZE]) {
            asm("" ::
                    : "memory");
        }
        barrier->flag[thread_id][(num_threads - 1) * CACHE_LINE_SIZE] = 0;
    }
    for (mask >>= 1; mask != 0; mask >>= 1) {
        barrier->flag[thread_id - mask][(num_threads - 1) * CACHE_LINE_SIZE] = 1;
    }
}

/* Initialize Tree barrier using POSIX condition variable */
void Tree_Posix_CV_Init(Tree_Posix_CV_t *barrier, int num_threads) {
    barrier->cv = (pthread_cond_t **)malloc(sizeof(pthread_cond_t *) * num_threads);
    for (int i = 0; i < num_threads; i++) {
        barrier->cv[i] = (pthread_cond_t *)malloc(sizeof(pthread_cond_t) * num_threads);
        for (int j = 0; j < num_threads; j++) {
            pthread_cond_init(&barrier->cv[i][j], NULL);
        }
    }
    barrier->mutex = (pthread_mutex_t **)malloc(sizeof(pthread_mutex_t *) * num_threads);
    for (int i = 0; i < num_threads; i++) {
        barrier->mutex[i] = (pthread_mutex_t *)malloc(sizeof(pthread_mutex_t) * num_threads);
        for (int j = 0; j < num_threads; j++) {
            pthread_mutex_init(&barrier->mutex[i][j], NULL);
        }
    }

    barrier->flag = (int **)malloc(sizeof(int *) * num_threads);
    for (int i = 0; i < num_threads; i++) {
        barrier->flag[i] = (int *)malloc(sizeof(int) * num_threads);

        for (int j = 0; j < num_threads; j++) {
            barrier->flag[i][j] = 0;
        }
    }
}

/* Tree barrier using POSIX condition variable */
void Tree_Posix_CV_Wait(Tree_Posix_CV_t *barrier, int thread_id, int num_threads) {
    unsigned int i, mask;
    for (i = 0, mask = 1; (mask & thread_id) != 0; i++, mask <<= 1) {
        pthread_mutex_lock(&barrier->mutex[thread_id][i]);
        while (!barrier->flag[thread_id][i]) {
            pthread_cond_wait(&barrier->cv[thread_id][i], &barrier->mutex[thread_id][i]);
            asm("" ::
                    : "memory");
        }
        pthread_mutex_unlock(&barrier->mutex[thread_id][i]);
        barrier->flag[thread_id][i] = 0;
    }

    if (thread_id < num_threads - 1) {
        barrier->flag[thread_id + mask][i] = 1;
        pthread_mutex_lock(&barrier->mutex[thread_id + mask][i]);
        pthread_cond_broadcast(&barrier->cv[thread_id + mask][i]);
        pthread_mutex_unlock(&barrier->mutex[thread_id + mask][i]);

        pthread_mutex_lock(&barrier->mutex[thread_id][num_threads - 1]);
        while (!barrier->flag[thread_id][num_threads - 1]) {
            pthread_cond_wait(&barrier->cv[thread_id][num_threads - 1], &barrier->mutex[thread_id][num_threads - 1]);
            asm("" ::
                    : "memory");
        }
        pthread_mutex_unlock(&barrier->mutex[thread_id][num_threads - 1]);

        barrier->flag[thread_id][num_threads - 1] = 0;
    }
    for (mask >>= 1; mask != 0; mask >>= 1) {
        // printf("asa %d\n", thread_id);
        barrier->flag[thread_id - mask][num_threads - 1] = 1;
        pthread_mutex_lock(&barrier->mutex[thread_id - mask][num_threads - 1]);
        pthread_cond_broadcast(&barrier->cv[thread_id - mask][num_threads - 1]);
        pthread_mutex_unlock(&barrier->mutex[thread_id - mask][num_threads - 1]);
    }
}
