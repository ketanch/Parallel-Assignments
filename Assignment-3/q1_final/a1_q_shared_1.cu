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

// #define THREADS_PER_BLOCK 32
#define TILE_SIZE 16
#define THREADS_PER_BLOCK_X TILE_SIZE
#define THREADS_PER_BLOCK_Y TILE_SIZE
#define THREADS_PER_BLOCK (THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_Y)

#define TOL 1e-5
#define ITER_LIMIT 1000

int nthreads, n;

__device__ float diff;
__managed__ int flag = 0;

bool checkEvenPower(long long int N) {
    if ((N & (N - 1)) != 0)
        return false;

    N = N & 0x55555555;

    return (N > 0);
}

__global__ void init_kernel(float *A, curandState_t *states, unsigned int seed, int n) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= (n + 2) * (n + 2)) {
        return;
    }
    curand_init(seed, id, 0, &states[id]);
    A[id] = curand_uniform(&states[id]);
}

__global__ void solve(float *A, int n, int span) {
    int done = 0, iters = 0;
    float temp, local_diff;
    cg::grid_group grid = cg::this_grid();
    while (!done) {
        local_diff = 0.0;
        if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
            diff = 0.0;
        }

        grid.sync();
        int outerTileIdx = blockIdx.x;
        int outerTileIdy = blockIdx.y;
        __shared__ float as[TILE_SIZE + 2][TILE_SIZE + 2];

        for (int k = 0; k < span; k++) {
            for (int l = 0; l < span; l++) {
                int tileIdx = outerTileIdx * span + k;
                int tileIdy = outerTileIdy * span + l;
                int idx = tileIdx * THREADS_PER_BLOCK_X + threadIdx.x + 1;
                int idy = tileIdy * THREADS_PER_BLOCK_Y + threadIdx.y + 1;

                as[threadIdx.x + 1][threadIdx.y + 1] = A[idx * (n + 2) + idy];
                if (threadIdx.x == 0) {
                    as[threadIdx.x][threadIdx.y + 1] = A[(idx - 1) * (n + 2) + idy];
                }
                if (threadIdx.y == 0) {
                    as[threadIdx.x + 1][threadIdx.y] = A[idx * (n + 2) + idy - 1];
                }
                if (threadIdx.x == TILE_SIZE - 1) {
                    as[threadIdx.x + 2][threadIdx.y + 1] = A[(idx + 1) * (n + 2) + idy];
                }
                if (threadIdx.y == TILE_SIZE - 1) {
                    as[threadIdx.x + 1][threadIdx.y + 2] = A[idx * (n + 2) + (idy + 1)];
                }
                temp = as[threadIdx.x + 1][threadIdx.y + 1];
                __syncthreads();
                A[idx * (n + 2) + idy] = 0.2 * (as[threadIdx.x + 1][threadIdx.y + 1] + as[threadIdx.x + 1][threadIdx.y] + as[threadIdx.x + 1][threadIdx.y + 2] + as[threadIdx.x + 2][threadIdx.y + 1] + as[threadIdx.x][threadIdx.y + 1]);
                local_diff += fabs(A[idx * (n + 2) + idy] - temp);
                __syncthreads();
            }
        }
        atomicAdd(&diff, local_diff);

        grid.sync();
        iters++;

        if (((diff / (n * n) < TOL) || (iters == ITER_LIMIT))) {
            done = 1;
            if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
                printf("[%d] diff = %.10f\n", iters, diff / (n * n));
            }
        }

        // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        //     printf("[%d] diff = %.10f\n", iters, diff / (n * n));
        // }
        grid.sync();
    }
}

__global__ void solve1(float *A, int n, int span) {
    int done = 0, iters = 0;
    float temp, local_diff;
    cg::grid_group grid = cg::this_grid();
    while (!done) {
        local_diff = 0.0;
        if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
            diff = 0.0;
        }

        grid.sync();
        int outerTileIdx = blockIdx.x;
        int outerTileIdy = blockIdx.y;
        __shared__ float as[TILE_SIZE + 2][TILE_SIZE + 2];

        int currOff = blockIdx.z;

        for (int k = 0; k < span; k++) {
            currOff = 1 - currOff;
            for (int l = currOff; l < span; l += gridDim.z) {
                int tileIdx = outerTileIdx * span + k;
                int tileIdy = outerTileIdy * span + l;
                int idx = tileIdx * THREADS_PER_BLOCK_X + threadIdx.x + 1;
                int idy = tileIdy * THREADS_PER_BLOCK_Y + threadIdx.y + 1;
                as[threadIdx.x + 1][threadIdx.y + 1] = A[idx * (n + 2) + idy];
                if (threadIdx.x == 0) {
                    as[threadIdx.x][threadIdx.y + 1] = A[(idx - 1) * (n + 2) + idy];
                }
                if (threadIdx.y == 0) {
                    as[threadIdx.x + 1][threadIdx.y] = A[idx * (n + 2) + idy - 1];
                }
                if (threadIdx.x == TILE_SIZE - 1) {
                    as[threadIdx.x + 2][threadIdx.y + 1] = A[(idx + 1) * (n + 2) + idy];
                }
                if (threadIdx.y == TILE_SIZE - 1) {
                    as[threadIdx.x + 1][threadIdx.y + 2] = A[idx * (n + 2) + (idy + 1)];
                }
                temp = as[threadIdx.x + 1][threadIdx.y + 1];
                __syncthreads();
                A[idx * (n + 2) + idy] = 0.2 * (as[threadIdx.x + 1][threadIdx.y + 1] + as[threadIdx.x + 1][threadIdx.y] + as[threadIdx.x + 1][threadIdx.y + 2] + as[threadIdx.x + 2][threadIdx.y + 1] + as[threadIdx.x][threadIdx.y + 1]);
                local_diff += fabs(A[idx * (n + 2) + idy] - temp);
                __syncthreads();
            }
        }
        atomicAdd(&diff, local_diff);

        grid.sync();
        iters++;

        if (((diff / (n * n) < TOL) || (iters == ITER_LIMIT))) {
            done = 1;
            if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
                printf("[%d] diff = %.10f\n", iters, diff / (n * n));
            }
        }

        // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
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
    // n = 1024;
    // nthreads = 1024;

    cudaMallocManaged((void **)&A, sizeof(float) * (n + 2) * (n + 2));

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
    // return 0;
    if (!checkEvenPower(nthreads)) {
        nthreads /= 2;
        flag = 1;
    }
    nthreads = round(sqrt(nthreads));
    printf("nthreads: %d\n", nthreads);

    curandState_t *states;
    cudaMalloc((void **)&states, (n + 2) * (n + 2) * sizeof(curandState_t));

    init_kernel<<<(((n + 2) * (n + 2)) / 1024) + ((n + 2) * (n + 2)) % 1024, 1024>>>(A, states, time(0), n);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();  // Get error code
    if (err != cudaSuccess) {
        printf("CUDA Error during initialization: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    int num_elements_per_thread = n / nthreads;
    int supportsCoopLaunch = 0;
    cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, device);
    void *kernelArgs[] = {(void *)&A, (void *)&n, (void *)&num_elements_per_thread};

    dim3 dimBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 dimGrid(nthreads / THREADS_PER_BLOCK_X, nthreads / THREADS_PER_BLOCK_Y);
    if (flag) {
        dimGrid.z = 2;
    }

    gettimeofday(&tv0, &tz0);
    if (flag) {
        cudaLaunchCooperativeKernel((void *)solve1, dimGrid, dimBlock, kernelArgs);
    } else {
        cudaLaunchCooperativeKernel((void *)solve, dimGrid, dimBlock, kernelArgs);
    }
    cudaDeviceSynchronize();

    gettimeofday(&tv1, &tz1);
    err = cudaGetLastError();  // Get error code

    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    printf("Time: %lf seconds\n", (double)((tv1.tv_sec - tv0.tv_sec) * 1000000 + (tv1.tv_usec - tv0.tv_usec)) / 1000000);

    return 0;
}
