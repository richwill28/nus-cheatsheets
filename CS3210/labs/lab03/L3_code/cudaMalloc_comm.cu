/**
 * Global Memory (Linear Array)
 * Demonstrates:
 * - Allocation of linear array by host
 * - Passing global memory pointer to device
 * - Method in which host accesses global memory
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
    int *start, *device_mem;
    int i, num_elements;

    cudaError_t rc;

    // Ask user for number of elements
    printf("How many elements to increment? ");
    scanf("%d", &num_elements);

    // Seed our RNG
    srand(0);

    // Malloc host memory
    start = (int*)malloc(num_elements * sizeof(int));
    // "Malloc" device memory
    cudaMalloc((void **)&device_mem, num_elements * sizeof(int));

    printf("Incrementor input:\n");
    for (i = 0; i < num_elements; i++) {
        start[i] = rand() % 100;
        printf("start[%d] = %d\n", i, start[i]);
    }

    /**
     * Copy values from start to our CUDA memory
     */
    rc = cudaMemcpy(device_mem, start, num_elements * sizeof(int), cudaMemcpyHostToDevice);

    if (rc != cudaSuccess)
    {
        printf("Could not copy to device. Reason: %s\n", cudaGetErrorString(rc));
    }

    incrementor<<<1, num_elements>>>(device_mem);
    check_cuda_errors();

    // Retrieve data from global memory
    rc = cudaMemcpy(start, device_mem, num_elements * sizeof(int), cudaMemcpyDeviceToHost);

    if (rc != cudaSuccess)
    {
        printf("Could not copy from device. Reason: %s\n", cudaGetErrorString(rc));
    }

    printf("Incrementor results:\n");
    for (i = 0; i < num_elements; i++) {
        printf("result[%d] = %d\n", i, start[i]);
    }

    // Free both host and device memory
    free(start);
    cudaFree(device_mem);

    return 0;
}
