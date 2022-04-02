#include <stdio.h>
#include <stdlib.h>

#define CACHE_LINE_SIZE 64

int max(int *arr, int len) {
    int max = arr[0];
    for (int i = 1; i < len; i++) {
        if (max < arr[i])
            max = arr[i]; 
    }
    return max;
}

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

int TestAndSet(int *addr) {
    int old_val = 1;

    /*If mem operand associated with xchg then no lock is required
     * ~ Intel manual
    */
    asm (
        "xchgl %0, %1 \n\t"
        : "+r" (old_val), "+m" (*addr) ::
    );

    return old_val;
}

void FetchAndInc(int val, int *addr) {}

void Init_Lamport_Bakery(int nthread, int *choosing, int *ticket) {
    ticket = (int*) malloc(nthread * CACHE_LINE_SIZE);
    for (int i = 0; i < nthread; i++) {
        ticket[i * CACHE_LINE_SIZE] = 0;
    }

    choosing = (int*) malloc(nthread * CACHE_LINE_SIZE);
    for (int i = 0; i < nthread; i++) {
        choosing[i * CACHE_LINE_SIZE] = 0;
    }
}

void Acquire_Lamport_Bakery(int pid, int *choosing, int *ticket) {
    choosing[pid * CACHE_LINE_SIZE] = 1;
    ticket[pid * CACHE_LINE_SIZE] = max(ticket, pid) + 1;
    choosing[pid * CACHE_LINE_SIZE] = 0;
    for (int i = 0; i < pid; i++) {
        while (choosing[i * CACHE_LINE_SIZE]);
        int ticket_i = ticket[i * CACHE_LINE_SIZE];
        int ticket_pid = ticket[pid * CACHE_LINE_SIZE];
        while (ticket_i && (ticket_i < ticket_pid || (ticket_i == ticket_pid && i < pid)));
    }
}

void Release_Lamport_Bakery(int pid, int *choosing, int *ticket) {
    ticket[pid * CACHE_LINE_SIZE] = 0;
}

void Acquire_Spinlock(int *lock_addr) {
    while (!CompareAndSet(0, 1, lock_addr));
}

void Release_Spinlock(int *lock_addr) {
    *lock_addr = 0;
}

void Acquire_TTS(int *addr) {
Lock:
    int reg_val = TestAndSet(addr);
    if (reg_val == 0) return;
    while (*addr != 0);
    goto Lock;
}

void Release_TTS(int *addr) {
    *addr = 0;
}

void Acquire_Ticket_Lock() {}

void Release_Ticket_Lock() {}