#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define ARRAYSIZE 1024 * 16
#define BLOCKSIZE 256


// One thread per output row over some N blocks
__global__ void multiplyKernel(float *c, const float *a, const float *b) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

	c[index] = 0;

	for (int j = 0; j < ARRAYSIZE; ++j)
		c[index] += a[index * ARRAYSIZE + j] * b[index];
}


// Helper function for using CUDA to multiply a matrix by a vector in parallel.
cudaError_t multiplyWithCuda(float *c, const float *a, const float *b)
{
    float *dev_a = 0;
    float *dev_b = 0;
    float *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, ARRAYSIZE * sizeof(float));
    cudaStatus = cudaMalloc((void**)&dev_a, ARRAYSIZE * ARRAYSIZE * sizeof(float));
    cudaStatus = cudaMalloc((void**)&dev_b, ARRAYSIZE * sizeof(float));

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, ARRAYSIZE * ARRAYSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_b, b, ARRAYSIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Launch a kernel on the GPU with one thread for each element.
    multiplyKernel<<<ARRAYSIZE/BLOCKSIZE, BLOCKSIZE>>>(dev_c, dev_a, dev_b);
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaDeviceSynchronize();

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, ARRAYSIZE * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

int main()
{
    float* a;
    float b[ARRAYSIZE];
    float c[ARRAYSIZE] = { 0 };

    int nIter = 1;

    // Allocate arraySize x arraySize input matrix a
    a = (float*)malloc(sizeof(float) * ARRAYSIZE * ARRAYSIZE);

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

    free(a);

    return 0;
}

