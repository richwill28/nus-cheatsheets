/**
 * Prints Thread ID of each thread
 * What to observe/ponder:
 * - Any trends on how the thread IDs are printed?
 * - Why are they printed like so?
 */

#include <stdio.h>

void check_cuda_errors()
{
    cudaError_t rc;
    rc = cudaGetLastError();
    if (rc != cudaSuccess)
    {
        printf("Last CUDA error %s\n", cudaGetErrorString(rc));
    }
}

__global__ void printer()
{
    printf("%d\n", threadIdx.x);
}

int main(int argc, char **argv)
{
    printer<<<1, 1024>>>();
    // Waits for all CUDA threads to complete.
    cudaDeviceSynchronize();
    check_cuda_errors();
    return 0;
}
