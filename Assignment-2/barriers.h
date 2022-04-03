typedef struct Central_Sense_Reversing_t {
    int counter;
    int flag;
    pthread_mutex_t mutex;
} Central_Sense_Reversing_t;

typedef struct Central_Posix_CV_t {
    int counter;
    pthread_cond_t cv;
    pthread_mutex_t mutex;
} Central_Posix_CV_t;

typedef struct Tree_Sense_Reversing_t {
    int **flag;
} Tree_Sense_Reversing_t;

// typedef struct Tree_Posix_CV_t {
//     pthread_cond_t **cv;
//     pthread_mutex_t *mutex;
// } Tree_Posix_CV_t;

void Central_Sense_Reversing_Init(Central_Sense_Reversing_t *barrier);
void Central_Sense_Reversing_Wait(Central_Sense_Reversing_t *barrier, int *localsense, int num_threads);
void Central_Posix_CV_Init(Central_Posix_CV_t *barrier);
void Central_Posix_CV_Wait(Central_Posix_CV_t *barrier, int num_threads);
void Tree_Sense_Reversing_Init(Tree_Sense_Reversing_t *barrier, int num_threads);
void Tree_Sense_Reversing_Wait(Tree_Sense_Reversing_t *barrier, int thread_id, int num_threads);
void Tree_Posix_CV_Init(Tree_Posix_CV_t *barrier, int num_threads);
void Tree_Posix_CV_Wait(Tree_Posix_CV_t *barrier, int thread_id, int num_threads);
