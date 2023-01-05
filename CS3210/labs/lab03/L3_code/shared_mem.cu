/**
 * Adds up 1,000,000 times of the block ID to
 * a variable.
 * What to observe/ponder:
 * - Any difference between shared and global memory?
 * - Does the result differ between runs?
 */

#include <stdio.h>

__device__ __managed__ volatile int global_counter[2];

void check_cuda_errors()
{
    cudaError_t rc;
    rc = cudaGetLastError();
    if (rc != cudaSuccess)
    {
        printf("Last CUDA error %s\n", cudaGetErrorString(rc));
    }
}

__global__ void shared_mem(int times)
{
    __shared__ int shared_counter[2];
    int i;

    // Zero out both counters
    shared_counter[threadIdx.x] = 0;

    for (i = 0; i < times; i++)
    {
        shared_counter[threadIdx.x] += blockIdx.x;
    }

    printf("Shared (Blk: %d, Th: %d): %d\n", blockIdx.x, threadIdx.x, shared_counter[threadIdx.x]);
}

__global__ void global_mem(int times)
{
    int i;

    // Zero out both counters
    global_counter[threadIdx.x] = 0;

    for (i = 0; i < times; i++)
    {
        global_counter[threadIdx.x] += blockIdx.x;
    }

    printf("Global (Blk: %d, Th: %d): %d\n", blockIdx.x, threadIdx.x, global_counter[threadIdx.x]);
}

int main(int argc, char **argv)
{
    shared_mem<<<10, 2>>>(1000000);
    cudaDeviceSynchronize();
    check_cuda_errors();

    global_mem<<<10, 2>>>(1000000);
    cudaDeviceSynchronize();
    check_cuda_errors();

    return 0;
}
