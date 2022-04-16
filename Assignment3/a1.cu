#include <assert.h>
#include <cooperative_groups.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
namespace cg = cooperative_groups;

#define THREADS_PER_BLOCK 1024
#define TOL 1e-5
#define ITER_LIMIT 1000

int nthreads, n;

__device__ int diff;

__global__ void init_kernel(float *A) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    // printf("%d\n", id);
    curandState_t state;

    curand_init(clock64(), id, 0, &state);

    A[id] = curand_uniform(&state);
}

__global__ void solve(float *A, int n, int nthreads) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    printf("%d\n", id);
    int done = 0, iters = 0;
    float temp, local_diff;
    while (!done) {
        local_diff = 0.0;
        if (id == 0) {
            diff = 0.0;
        }
        cg::grid_group grid = cg::this_grid();
        grid.sync();
        for (int i = id * (n / nthreads) + 1; i < ((id + 1) * (n / nthreads)) + 1; i++) {
            for (int j = 1; j < n + 1; j++) {
                temp = A[i * n + j];
                A[i * n + j] = 0.2 * (A[i * n + j] + A[i * n + j - 1] + A[i * n + j + 1] + A[(i + 1) * n + j] + A[(i - 1) * n + j]);
                local_diff += fabs(A[i * n + j] - temp);
            }
        }
        atomicAdd(&diff, local_diff);
        grid.sync();
        iters++;
        if ((diff / (n * n) < TOL) || (iters == ITER_LIMIT)) {
            printf("%d\n", iters);
            done = 1;
        }
        grid.sync();

        // #pragma omp master
        //         fprintf(fp, "[%d] diff = %.10f\n", iters, diff / (n * n));
    }
}

int main(int argc, char **argv) {
    struct timeval tv0, tv1;
    struct timezone tz0, tz1;
    float *A;

    if (argc != 3) {
        printf("Need grid size (n) and number of threads (nthreads).\nAborting...\n");
        exit(1);
    }

    n = atoi(argv[1]);
    nthreads = atoi(argv[2]);

    cudaMallocManaged((void **)&A, sizeof(float) * (n) * (n));

    int numBlocksPerSm = 0, numBlocks;
    cudaDeviceProp deviceProp;
    int device = -1;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&deviceProp, device);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, solve, THREADS_PER_BLOCK, 0);
    numBlocks = deviceProp.multiProcessorCount * numBlocksPerSm;
    printf("Max number of blocks per SM: %d, number of SMs: %d, number of blocks: %d\n", numBlocksPerSm, deviceProp.multiProcessorCount, numBlocks);
    while ((numBlocks & (numBlocks - 1)) != 0) numBlocks--;
    if (nthreads > (THREADS_PER_BLOCK * numBlocks)) nthreads = THREADS_PER_BLOCK * numBlocks;
    printf("Number of blocks: %d, Threads per block: %d, Total number of threads: %d\n", nthreads / THREADS_PER_BLOCK, THREADS_PER_BLOCK, nthreads);

    init_kernel<<<(n * n) / 1024, 1024>>>(A);
    cudaDeviceSynchronize();
    printf("fdsdfsdk\n\n\n");

    // int num_elements_per_thread = (n * n) / nthreads;
    int supportsCoopLaunch = 0;
    cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, device);
    void *kernelArgs[] = {(void *)&A, (void *)&n, (void *)&nthreads};
    dim3 dimBlock(THREADS_PER_BLOCK, 1, 1);
    dim3 dimGrid(nthreads / THREADS_PER_BLOCK, 1, 1);
    printf("fdsdfsdk\n\n\n test:  %d\n", supportsCoopLaunch);

    gettimeofday(&tv0, &tz0);
    cudaLaunchCooperativeKernel((void *)solve, dimGrid, dimBlock, kernelArgs);
    cudaDeviceSynchronize();

    // for (int i = 0; i < n; i++) {
    //     for (int j = 0; j < n; j++) {
    //         assert(A[i * n + j] > 1);
    //         // printf("%f ", A[i * n + j]);
    //     }
    //     // printf("\n");
    // }

    // #pragma omp parallel num_threads(nthreads)
    //     Solve(fp);

    gettimeofday(&tv1, &tz1);

    printf("Time: %lf seconds\n", (double)((tv1.tv_sec - tv0.tv_sec) * 1000000 + (tv1.tv_usec - tv0.tv_usec)) / 1000000);

    return 0;
}
