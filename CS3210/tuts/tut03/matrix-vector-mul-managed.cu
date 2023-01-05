#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define ARRAYSIZE 1024 * 16
#define BLOCKSIZE 512

__managed__ float *a;
__managed__ float *b;
__managed__ float *c;
 


// One thread per output row over some N blocks
__global__ void multiplyKernel() {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

	c[index] = 0;

	for (int j = 0; j < ARRAYSIZE; ++j)
		c[index] += a[index * ARRAYSIZE + j] * b[index];
}


// Helper function for using CUDA to multiply a matrix by a vector in parallel.
void multiplyWithCuda(float *c, const float *a, const float *b)
{
    // Launch a kernel on the GPU with one thread for each element.
    multiplyKernel<<<ARRAYSIZE/BLOCKSIZE, BLOCKSIZE>>>();
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaDeviceSynchronize();
}

int main()
{
    // float* a;
    // float b[ARRAYSIZE];
    // float c[ARRAYSIZE] = { 0 };

    int nIter = 1;

    // Allocate arraySize x arraySize input matrix a
    cudaMallocManaged(&a, sizeof(float) * ARRAYSIZE * ARRAYSIZE);
    cudaMallocManaged(&b, sizeof(float) * ARRAYSIZE);
    cudaMallocManaged(&c, sizeof(float) * ARRAYSIZE);

    // Initialize a
    for (int i = 0; i < ARRAYSIZE; ++i) {
	    for (int j = 0; j < ARRAYSIZE; ++j) {
			a[i* ARRAYSIZE + j] = (float)(i * j);
		}
    }
    // Initialize b
    for (int i = 0; i < ARRAYSIZE; ++i) {
		b[i] = (float)i;
    }
    // Initialize c
    for (int i = 0; i < ARRAYSIZE; ++i) {
		c[i] = 0;
    }

    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start;
    cudaEventCreate(&start);

    cudaEvent_t stop;
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, NULL);

	// Execute the kernel
	for (int j = 0; j < nIter; j++) {
		multiplyWithCuda(c, a, b);
	}

    // Record the stop event
    cudaEventRecord(stop, NULL);

    // Wait for the stop event to complete
    cudaEventSynchronize(stop);

    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);

    printf("Time= %.3f msec\n", msecTotal);

    return 0;
}

