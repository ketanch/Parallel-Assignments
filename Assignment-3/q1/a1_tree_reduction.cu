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

#define THREADS_PER_BLOCK 32
#define TOL 1e-5
#define ITER_LIMIT 1000

int nthreads, n;

__device__ float diff;

__global__ void init_kernel(float *A, curandState_t *states, unsigned int seed, int n) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= (n + 2) * (n + 2)) {
        return;
    }
    curand_init(seed, id, 0, &states[id]);
    A[id] = curand_uniform(&states[id]);
}

__global__ void solve(float *A, int n, int span) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int done = 0, iters = 0;
    float temp, local_diff;
    cg::grid_group grid = cg::this_grid();
    while (!done) {
        local_diff = 0.0;
        if (id == 0) {
            diff = 0.0;
        }

        grid.sync();

        for (int z = 0; z < span; z++) {
            int currInd = id * span + z;
            int i = currInd / n;
            int j = currInd % n;
            i += 1;
            j += 1;

            temp = A[i * n + j];

            A[i * n + j] = 0.2 * (A[i * n + j] + A[i * n + j - 1] + A[i * n + j + 1] + A[(i + 1) * n + j] + A[(i - 1) * n + j]);

            local_diff += fabs(A[i * n + j] - temp);
        }
        unsigned mask = 0xffffffff;
        for (int i = warpSize / 2; i > 0; i = i / 2) {
            local_diff += __shfl_down_sync(mask, local_diff, i);
        }
        if (threadIdx.x % warpSize == 0) {
            atomicAdd(&diff, local_diff);
        }

        grid.sync();
        iters++;

        if (((diff / (n * n) < TOL) || (iters == ITER_LIMIT))) {
            done = 1;
            // printf("id: %d, iters: %d\n", id, iters);
            // grid.sync();
        }

        // if (id == 0) {
        //     printf("[%d] diff = %.10f\n", iters, diff / (n * n));
        // }
        grid.sync();
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
    // n = 512;
    // nthreads = 4096;
    int device = -1;
    cudaGetDevice(&device);
    cudaMallocManaged((void **)&A, sizeof(float) * (n + 2) * (n + 2));
    cudaMemAdvise(&A, sizeof(float) * (n + 2) * (n + 2), cudaMemAdviseSetPreferredLocation, device);

    int numBlocksPerSm = 0, numBlocks;
    cudaDeviceProp deviceProp;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&deviceProp, device);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, solve, THREADS_PER_BLOCK, 0);
    numBlocks = deviceProp.multiProcessorCount * numBlocksPerSm;
    printf("Max number of blocks per SM: %d, number of SMs: %d, number of blocks: %d\n", numBlocksPerSm, deviceProp.multiProcessorCount, numBlocks);
    while ((numBlocks & (numBlocks - 1)) != 0) numBlocks--;
    if (nthreads > (THREADS_PER_BLOCK * numBlocks)) nthreads = THREADS_PER_BLOCK * numBlocks;
    printf("Number of blocks: %d, Threads per block: %d, Total number of threads: %d\n", nthreads / THREADS_PER_BLOCK, THREADS_PER_BLOCK, nthreads);

    curandState_t *states;
    cudaMalloc((void **)&states, (n + 2) * (n + 2) * sizeof(curandState_t));

    init_kernel<<<(((n + 2) * (n + 2)) / 1024) + ((n + 2) * (n + 2)) % 1024, 1024>>>(A, states, time(0), n);
    cudaDeviceSynchronize();

    int num_elements_per_thread = (n * n) / nthreads;
    int supportsCoopLaunch = 0;
    cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, device);
    void *kernelArgs[] = {(void *)&A, (void *)&n, (void *)&num_elements_per_thread};
    dim3 dimBlock(THREADS_PER_BLOCK, 1, 1);
    dim3 dimGrid(nthreads / THREADS_PER_BLOCK, 1, 1);

    gettimeofday(&tv0, &tz0);
    cudaLaunchCooperativeKernel((void *)solve, dimGrid, dimBlock, kernelArgs);
    cudaDeviceSynchronize();

    gettimeofday(&tv1, &tz1);

    // for (int i = 0; i < n; i++) {
    //     for (int j = 0; j < n; j++) {
    //         printf("%f ", A[i * n + j]);
    //     }
    //     printf("\n");
    //     // break;
    // }

    printf("Time: %lf seconds\n", (double)((tv1.tv_sec - tv0.tv_sec) * 1000000 + (tv1.tv_usec - tv0.tv_usec)) / 1000000);

    return 0;
}
