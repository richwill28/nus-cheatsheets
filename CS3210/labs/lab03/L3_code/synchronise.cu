/**
 * Demonstrates the use of a synchronisation construct
 * What to ponder:
 * - Significance of counter values printed out with/without syncthreads
 * - Why does the values vary when syncthreads is used in a kernel launch
 *   containing multiple blocks?
 */

#include <stdio.h>

__device__ __managed__ int volatile is_done;
__device__ __managed__ int counter;

void check_cuda_errors()
{
    cudaError_t rc;
    rc = cudaGetLastError();
    if (rc != cudaSuccess)
    {
        printf("Last CUDA error %s\n", cudaGetErrorString(rc));
    }

}

__global__ void no_sync()
{
    // Block till our first warp passes atomicAdd
    while (threadIdx.x / 32 != 0 && is_done == 0) { }

    atomicAdd(&counter, 1);
    if (threadIdx.x == 0)
    {
        is_done = 1;
    }

    if (threadIdx.x == 0)
    {
        printf("Counter value (no sync): %d\n", counter);
    }
}

__global__ void with_sync()
{
    // Block till our first warp passes atomicAdd
    while (threadIdx.x / 32 != 0 && is_done == 0) { }

    atomicAdd(&counter, 1);

    if (threadIdx.x == 0)
    {
        is_done = 1;
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        printf("Counter value (sync, one block): %d\n", counter);
    }
}

__global__ void with_sync_multiple()
{
    // Block till our first warp passes atomicAdd
    while (blockIdx.x / 32 != 0 && threadIdx.x / 32 != 0 && is_done == 0) { }

    atomicAdd(&counter, 1);

    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        is_done = 1;
    }

    __syncthreads();

    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        printf("Counter value (sync, multiple blocks): %d\n", counter);
    }
}

int main(int argc, char **argv)
{
    is_done = 0;
    counter = 0;

    no_sync<<<1, 1024>>>();
    cudaDeviceSynchronize();
    check_cuda_errors();

    is_done = 0;
    counter = 0;

    with_sync<<<1, 1024>>>();
    cudaDeviceSynchronize();
    check_cuda_errors();

    is_done = 0;
    counter = 0;

    with_sync_multiple<<<1024, 512>>>();
    cudaDeviceSynchronize();
    check_cuda_errors();

    return 0;
}
