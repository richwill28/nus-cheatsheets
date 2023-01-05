/**
 * Global Memory (Linear Array) using Unified Memory
 */
#include <stdio.h>
#include <stdlib.h>

void check_cuda_errors()
{
    cudaError_t rc;
    rc = cudaGetLastError();
    if (rc != cudaSuccess)
    {
        printf("Last CUDA error %s\n", cudaGetErrorString(rc));
    }

}

__global__ void incrementor(int* numbers)
{
    numbers[threadIdx.x]++;
}

int main(int argc, char **argv)
{
    int *device_mem;
    int i, num_elements;

    // Ask user for number of elements
    printf("How many elements to increment? ");
    scanf("%d", &num_elements);

    // Seed our RNG
    srand(0);

    // "Malloc" device memory
    cudaMallocManaged((void **)&device_mem, num_elements * sizeof(int));

    printf("Incrementor input:\n");
    for (i = 0; i < num_elements; i++) {
        device_mem[i] = rand() % 100;
        printf("start[%d] = %d\n", i, device_mem[i]);
    }

    incrementor<<<1, num_elements>>>(device_mem);
    check_cuda_errors();

    // Ensure that we don't proceed till we get the results!
    cudaDeviceSynchronize();

    printf("Incrementor results:\n");
    for (i = 0; i < num_elements; i++) {
        printf("result[%d] = %d\n", i, device_mem[i]);
    }

    // Free both host and device memory
    cudaFree(device_mem);

    return 0;
}
