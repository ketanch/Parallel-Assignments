#include <assert.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// #define THREADS_PER_BLOCK 32
#define TILE_SIZE 16
#define THREADS_PER_BLOCK_X TILE_SIZE
#define THREADS_PER_BLOCK_Y TILE_SIZE
// #define THREADS_PER_BLOCK (THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_Y)
#define THREADS_PER_BLOCK 128

int nthreads, n;

__device__ float diff;

bool checkEvenPower(long long int N) {
    if ((N & (N - 1)) != 0)
        return false;

    N = N & 0x55555555;

    return (N > 0);
}

__global__ void init_kernel(float *A, curandState_t *states, unsigned int seed, int n) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &states[id]);
    A[id] = curand_uniform(&states[id]);
}

__global__ void solve1(float *A, float *B, float *C, int n, int span) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    id = id * span;
    int idx = id % n;
    int idy = id / n;

    float temp = 0;
    int newRow = 0;
    int i;
    for (i = 0; i < span; i++) {
        if (((i + idx) % n) == 0 && i != 0) {
            atomicAdd(&C[idy + newRow], temp);
            newRow++;
            temp = 0;
        }
        temp += A[((i + idx) % n) + (idy + newRow) * n] * B[(i + idx) % n];
    }
    atomicAdd(&C[idy + newRow], temp);
}

int main(int argc, char **argv) {
    struct timeval tv0, tv1;
    struct timezone tz0, tz1;
    float *A, *B, *C;

    if (argc != 3) {
        printf("Need grid size (n) and number of threads (nthreads).\nAborting...\n");
        exit(1);
    }

    n = atoi(argv[1]);
    nthreads = atoi(argv[2]);
    // n = 1024;
    // nthreads = 1024;

    cudaMallocManaged((void **)&A, sizeof(float) * n * n);
    cudaMallocManaged((void **)&B, sizeof(float) * n);
    cudaMallocManaged((void **)&C, sizeof(float) * n);

    printf("nthreads: %d\n", nthreads);

    curandState_t *states;
    cudaMalloc((void **)&states, n * n * sizeof(curandState_t));

    init_kernel<<<(n * n) / 1024, 1024>>>(A, states, time(0), n);
    if (n >= 1024) {
        init_kernel<<<n / 1024, 1024>>>(B, states, time(0), n);
    } else {
        init_kernel<<<1, n>>>(B, states, time(0), n);
    }
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();  // Get error code
    if (err != cudaSuccess) {
        printf("CUDA Error during initialization: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    int num_elements_per_thread = (n * n) / nthreads;

    // dim3 dimBlock(THREADS_PER_BLOCK);
    // dim3 dimGrid(nthreads / THREADS_PER_BLOCK);
    gettimeofday(&tv0, &tz0);
    solve1<<<nthreads / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(A, B, C, n, num_elements_per_thread);
    // cudaLaunchCooperativeKernel((void *)solve1, dimGrid, dimBlock, kernelArgs);
    cudaDeviceSynchronize();

    gettimeofday(&tv1, &tz1);
    err = cudaGetLastError();  // Get error code

    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    float diff = 0;

    float *D = (float *)calloc(n, sizeof(float));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            // printf("%f ", A[i * n + j]);
            D[i] += A[i * n + j] * B[j];
        }
        // printf("\n");
        diff += abs(D[i] - C[i]);
    }

    printf("Average diff: %f\n", diff / n);

    printf("Time: %lf seconds\n", (double)((tv1.tv_sec - tv0.tv_sec) * 1000000 + (tv1.tv_usec - tv0.tv_usec)) / 1000000);

    return 0;
}
