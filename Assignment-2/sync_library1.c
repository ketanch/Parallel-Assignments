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

    asm (
        "lock cmpxchgl %4, %1 \n\t"
        "setzb %0"
        : "=qm" (result), "+m" (*addr), "=a" (old_val_out)
        : "a" (old_val), "r" (new_val) :
    );

    return result;
}

/*Implements test and set operation using xchg*/
int TestAndSet(int *addr) {
    int old_val = 1;

    asm (
        "lock xchgl %0, %1 \n\t"
        : "+r" (old_val), "+m" (*addr) ::
    );

    return old_val;
}

/*Implements fetch and increment using cmpxchg*/
int FetchAndInc(int *addr) {
GetLock:
    int old_val = *addr;
    int new_val = old_val + 1;
    int res = CompareAndSet(old_val, new_val, addr);
    if (!res)
        goto GetLock;
    
    return old_val;
    
}

/*Initializes arrays for Lamport Bakery*/
void Init_Lamport_Bakery(int nthread, int **choosing, int **ticket) {
    *ticket = (int*) malloc(nthread * CACHE_LINE_SIZE);
    for (int i = 0; i < nthread; i++) {
        (*ticket)[i * CACHE_LINE_SIZE] = 0;
    }
    *choosing = (int*) malloc(nthread * CACHE_LINE_SIZE);
    for (int i = 0; i < nthread; i++) {
        (*choosing)[i * CACHE_LINE_SIZE] = 0;
    }
}

/*Acquire for Lamport Bakery*/
void Acquire_Lamport_Bakery(int pid, int nthreads, int *choosing, int *ticket) {
    choosing[pid * CACHE_LINE_SIZE] = 1;
    asm ("mfence":::"memory");
    ticket[pid * CACHE_LINE_SIZE] = max(ticket, nthreads) + 1;
    asm ("mfence":::"memory");
    choosing[pid * CACHE_LINE_SIZE] = 0;
    asm ("mfence":::"memory");
    for (int i = 0; i < nthreads; i++) {
        while (choosing[i * CACHE_LINE_SIZE]) {
            asm ("":::"memory");
        }
        int ticket_i = ticket[i * CACHE_LINE_SIZE];
        int ticket_pid = ticket[pid * CACHE_LINE_SIZE];
        while (ticket_i && (ticket_i < ticket_pid || (ticket_i == ticket_pid && i < pid))) {
                ticket_i = ticket[i * CACHE_LINE_SIZE];
                ticket_pid = ticket[pid * CACHE_LINE_SIZE];
                asm ("":::"memory");
        }
    }
}

/*Release for Lamport Bakery*/
void Release_Lamport_Bakery(int pid, int *choosing, int *ticket) {
    asm ("":::"memory");
    ticket[pid * CACHE_LINE_SIZE] = 0;
}

/*Acquire for Spinlock*/
void Acquire_Spinlock(int *lock_addr) {
    while (!CompareAndSet(0, 1, lock_addr));
}

/*Release for Spinlock*/
void Release_Spinlock(int *lock_addr) {
    asm ("":::"memory");
    *lock_addr = 0;
}

/*Acquire for Test & test & set Lock*/
void Acquire_TTS(int *addr) {
Lock:
    int reg_val = TestAndSet(addr);
    if (reg_val == 0) return;
    while (*addr != 0) {
        asm ("":::"memory");
    }
    goto Lock;
}

/*Release for Test & test & set lock*/
void Release_TTS(int *addr) {
    asm ("":::"memory");
    *addr = 0;
}

/*Acquire for Ticket Lock*/
void Acquire_Ticket_Lock(int *ticket_addr, int *release_count_addr) {
    int ticket = FetchAndInc(ticket_addr);
    while (ticket != *release_count_addr) {
        asm ("":::"memory");
    }
    return;
}

/*Release for Ticket lock*/
void Release_Ticket_Lock(int *release_count_addr) {
    asm ("":::"memory");
    *release_count_addr += 1;
}

/*Initializes array for Array lock*/
char* Init_Array_Lock(int len) {
    char *arr = (char*) malloc(len * sizeof(char) * CACHE_LINE_SIZE);
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
        asm ("":::"memory");
    }
    return th_index;
}

/*Release for Array Lock*/
void Release_Array_Lock(char *lock_addr, int th_index, int arr_len) {
    asm ("":::"memory");
    lock_addr[th_index * CACHE_LINE_SIZE] = 0;
    lock_addr[((th_index + 1) % arr_len) * CACHE_LINE_SIZE] = 1;
}