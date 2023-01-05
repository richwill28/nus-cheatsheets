/**
 * Global Memory (Symbol) using Unified Memory
 */
#include <stdio.h>
#include <stdlib.h>

#define NUM_ELEMENTS 5

__managed__ int result[NUM_ELEMENTS];

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
    int i;

    // Seed our RNG
    srand(0);

    printf("Incrementor input:\n");
    for (i = 0; i < NUM_ELEMENTS; i++) {
        result[i] = rand() % 100;
        printf("start[%d] = %d\n", i, result[i]);
    }

    incrementor<<<1, NUM_ELEMENTS>>>();

    // Ensure that we don't proceed till we get the results!
    cudaDeviceSynchronize();
    check_cuda_errors();

    printf("Incrementor results:\n");
    for (i = 0; i < NUM_ELEMENTS; i++) {
        printf("result[%d] = %d\n", i, result[i]);
    }
    return 0;
}
