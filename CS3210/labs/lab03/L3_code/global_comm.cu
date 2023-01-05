/**
 * Global Memory (Symbol)
 * Demonstrates:
 * - Communication between host and device
 * - Method in which host accesses global memory
 */
#include <stdio.h>
#include <stdlib.h>

#define NUM_ELEMENTS 5

__device__ int result[NUM_ELEMENTS];

void check_cuda_errors()
{
    cudaError_t rc;
    rc = cudaGetLastError();
    if (rc != cudaSuccess)
    {
        printf("Last CUDA error %s\n", cudaGetErrorString(rc));
    }
}

__global__ void incrementor()
{
    result[threadIdx.x]++;
}

int main(int argc, char **argv)
{
    int start[NUM_ELEMENTS];
    int host_result[NUM_ELEMENTS];
    int i;

    cudaError_t rc;

    // Seed our RNG
    srand(0);

    printf("Incrementor input:\n");
    for (i = 0; i < NUM_ELEMENTS; i++)
    {
        start[i] = rand() % 100;
        printf("start[%d] = %d\n", i, start[i]);
    }

    /**
     * Copy a value from result to host
     */
    rc = cudaMemcpyToSymbol(result, &start, sizeof(start));

    if (rc != cudaSuccess)
    {
        printf("Could not copy to device. Reason: %s\n", cudaGetErrorString(rc));
    }

    incrementor<<<1, NUM_ELEMENTS>>>();
    check_cuda_errors();

    // Retrieve data from global memory variable
    rc = cudaMemcpyFromSymbol(&host_result, result, sizeof(start));

    if (rc != cudaSuccess)
    {
        printf("Could not copy from device. Reason: %s\n", cudaGetErrorString(rc));
    }

    for (int x = 0; x < NUM_ELEMENTS; x++)
    {
        printf("%d\n", result[x]);
    }

    printf("Incrementor results:\n");
    for (i = 0; i < NUM_ELEMENTS; i++)
    {
        printf("result[%d] = %d\n", i, host_result[i]);
    }
    return 0;
}
