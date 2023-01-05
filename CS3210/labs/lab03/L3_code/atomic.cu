/**
 * All threads increments a counter in global memory
 * by one. The difference is that one uses CUDA's atomic function
 * to perform an increment.
 * What to observe/ponder:
 * - What are the values that are printed out?
 * - Are they consistent across runs?
 */

#include <stdio.h>

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

__global__ void non_atomic()
{
    counter++;
}

__global__ void atomic()
{
    atomicAdd(&counter, 1);
}

int main(int argc, char **argv)
{
    // Set up counter
    counter = 0;

    dim3 gridDim(128, 128);
    dim3 blockDim(32, 32);
    non_atomic<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();
    check_cuda_errors();

    printf("Result from non-atomic increment by 16777216 threads: %d\n", counter);

    // Reset counter
    counter = 0;

    atomic<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();
    check_cuda_errors();

    printf("Result from atomic increment by 16777216 threads: %d\n", counter);

    return 0;
}
