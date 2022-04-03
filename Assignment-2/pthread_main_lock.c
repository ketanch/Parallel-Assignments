#include <assert.h>
#include <pthread.h>
#include <sys/time.h>
#include "locks.h"

#define N 10000000

int nthreads;
int x = 0, y = 0;

/*POSIX lock*/
pthread_mutex_t posix_lock;

/*Binary Semaphore*/
sem_t sema_lock;

/*Lamport Bakery data*/
int *choosing;
int *ticket;

/*Spinlock data*/
int spin_lock;

/*TTS data*/
int tts_lock;

/*Ticket lock data*/
int tl_ticket;
int release_count;

/*Array data*/
char *arr_lock;
int arr_len;
int arr_index;

/*Compute functions for each type of lock*/
/*-----------------------------------------*/
void *compute_posix(void *param) {
    int pid = *(int *)param;
    for (int i = 0; i < N; i++) {
        Acquire_pthread_mutex(&posix_lock);
        assert(x == y);
        x = y + 1;
        y++;
        Release_pthread_mutex(&posix_lock);
    }
}

void *compute_semaphore(void *param) {
    int pid = *(int *)param;
    for (int i = 0; i < N; i++) {
        Acquire_sema_lock(&sema_lock);
        assert(x == y);
        x = y + 1;
        y++;
        Release_sema_lock(&sema_lock);
    }
}

void *compute_lamport_bakery(void *param) {
    int pid = *(int *)param;
    for (int i = 0; i < N; i++) {
        Acquire_Lamport_Bakery(pid, nthreads, choosing, ticket);
        assert(x == y);
        x = y + 1;
        y++;
        Release_Lamport_Bakery(pid, choosing, ticket);
    }
}

void *compute_spinlock(void *param) {
    int pid = *(int *)param;
    for (int i = 0; i < N; i++) {
        Acquire_Spinlock(&spin_lock);
        assert(x == y);
        x = y + 1;
        y++;
        Release_Spinlock(&spin_lock);
    }
}

void *compute_tts(void *param) {
    int pid = *(int *)param;
    for (int i = 0; i < N; i++) {
        Acquire_TTS(&tts_lock);
        assert(x == y);
        x = y + 1;
        y++;
        Release_TTS(&tts_lock);
    }
}

void *compute_ticket_lock(void *param) {
    int pid = *(int *)param;
    for (int i = 0; i < N; i++) {
        Acquire_Ticket_Lock(&tl_ticket, &release_count);
        assert(x == y);
        x = y + 1;
        y++;
        Release_Ticket_Lock(&release_count);
    }
}

void *compute_array(void *param) {
    int pid = *(int *)param;
    for (int i = 0; i < N; i++) {
        int th_index = Acquire_Array_Lock(arr_lock, arr_len, &arr_index);
        assert(x == y);
        x = y + 1;
        y++;
        Release_Array_Lock(arr_lock, th_index, arr_len);
    }
}
/*-----------------------------------------*/

int main(int argc, char *argv[]) {
    int *id;
    pthread_t *tid;
    struct timeval tv0, tv1;
    struct timezone tz0, tz1;

    if (argc != 2) {
        printf("Need number of threads.\n");
        exit(1);
    }
    nthreads = atoi(argv[1]);

    /*Lock data initialization*/
    if (pthread_mutex_init(&posix_lock, NULL) != 0) {
        printf("Mutex initialization failed\n");
        exit(1);
    }
    sem_init(&sema_lock, 0, 1);
    Init_Lamport_Bakery(nthreads, &choosing, &ticket);
    spin_lock = 0;
    tts_lock = 0;
    tl_ticket = 0;
    release_count = 0;
    arr_len = nthreads;
    arr_lock = Init_Array_Lock(arr_len);
    arr_index = 0;
    /*-----------------------------------------*/

    tid = (pthread_t *)malloc(nthreads * sizeof(pthread_t));
    id = (int *)malloc(nthreads * sizeof(int));
    for (int i = 0; i < nthreads; i++)
        id[i] = i;

    /*Benchmarking POSIX mutex lock*/
    x = 0, y = 0;
    gettimeofday(&tv0, &tz0);
    for (int i = 0; i < nthreads; i++) {
        pthread_create(&tid[i], NULL, compute_posix, &id[i]);
    }

    for (int i = 0; i < nthreads; i++) {
        pthread_join(tid[i], NULL);
    }
    assert(x == y);
    assert(x == N * nthreads);
    gettimeofday(&tv1, &tz1);
    pthread_mutex_destroy(&posix_lock);
    printf("Lock = POSIX_Lock, time = %ld microseconds\n", (tv1.tv_sec - tv0.tv_sec) * 1000000 + (tv1.tv_usec - tv0.tv_usec));


    /*Benchmarking semaphore lock*/
    x = 0, y = 0;
    gettimeofday(&tv0, &tz0);
    for (int i = 0; i < nthreads; i++) {
        pthread_create(&tid[i], NULL, compute_semaphore, &id[i]);
    }

    for (int i = 0; i < nthreads; i++) {
        pthread_join(tid[i], NULL);
    }
    assert(x == y);
    assert(x == N * nthreads);
    gettimeofday(&tv1, &tz1);
    sem_destroy(&sema_lock);
    printf("Lock = Semaphore_Lock, time = %ld microseconds\n", (tv1.tv_sec - tv0.tv_sec) * 1000000 + (tv1.tv_usec - tv0.tv_usec));


    /*Benchmarking Lamport Lock*/
    x = 0, y = 0;
    gettimeofday(&tv0, &tz0);
    for (int i = 0; i < nthreads; i++) {
        pthread_create(&tid[i], NULL, compute_lamport_bakery, &id[i]);
    }

    for (int i = 0; i < nthreads; i++) {
        pthread_join(tid[i], NULL);
    }
    assert(x == y);
    assert(x == N * nthreads);
    gettimeofday(&tv1, &tz1);
    printf("Lock = Lamport_Lock, time = %ld microseconds\n", (tv1.tv_sec - tv0.tv_sec) * 1000000 + (tv1.tv_usec - tv0.tv_usec));

    /*Benchmarking SpinLock*/
    x = 0, y = 0;
    gettimeofday(&tv0, &tz0);
    for (int i = 0; i < nthreads; i++) {
        pthread_create(&tid[i], NULL, compute_spinlock, &id[i]);
    }

    for (int i = 0; i < nthreads; i++) {
        pthread_join(tid[i], NULL);
    }
    assert(x == y);
    assert(x == N * nthreads);
    gettimeofday(&tv1, &tz1);
    printf("Lock = Spinlock, time = %ld microseconds\n", (tv1.tv_sec - tv0.tv_sec) * 1000000 + (tv1.tv_usec - tv0.tv_usec));

    /*Benchmarking TTS lock*/
    x = 0, y = 0;
    gettimeofday(&tv0, &tz0);
    for (int i = 0; i < nthreads; i++) {
        pthread_create(&tid[i], NULL, compute_tts, &id[i]);
    }

    for (int i = 0; i < nthreads; i++) {
        pthread_join(tid[i], NULL);
    }
    assert(x == y);
    assert(x == N * nthreads);
    gettimeofday(&tv1, &tz1);
    printf("Lock = TTS_Lock, time = %ld microseconds\n", (tv1.tv_sec - tv0.tv_sec) * 1000000 + (tv1.tv_usec - tv0.tv_usec));

    /*Benchmarking Ticket Lock*/
    x = 0, y = 0;
    gettimeofday(&tv0, &tz0);
    for (int i = 0; i < nthreads; i++) {
        pthread_create(&tid[i], NULL, compute_ticket_lock, &id[i]);
    }

    for (int i = 0; i < nthreads; i++) {
        pthread_join(tid[i], NULL);
    }
    assert(x == y);
    assert(x == N * nthreads);
    gettimeofday(&tv1, &tz1);
    printf("Lock = Ticket_Lock, time = %ld microseconds\n", (tv1.tv_sec - tv0.tv_sec) * 1000000 + (tv1.tv_usec - tv0.tv_usec));

    /*Benchmarking Array Lock*/
    x = 0, y = 0;
    gettimeofday(&tv0, &tz0);
    for (int i = 0; i < nthreads; i++) {
        pthread_create(&tid[i], NULL, compute_array, &id[i]);
    }

    for (int i = 0; i < nthreads; i++) {
        pthread_join(tid[i], NULL);
    }
    assert(x == y);
    assert(x == N * nthreads);
    gettimeofday(&tv1, &tz1);
    printf("Lock = Array_Lock, time = %ld microseconds\n", (tv1.tv_sec - tv0.tv_sec) * 1000000 + (tv1.tv_usec - tv0.tv_usec));

    return 0;
}