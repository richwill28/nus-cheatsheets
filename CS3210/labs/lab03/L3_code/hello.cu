/*
 * Hello World in CUDA
 *
 * CS3210
 *
 * This program should print "HELLO WORLD" if successful.
 *
 */

#include <stdio.h>

#define N 32

// #define DISCRETE

// __global__ void hello(char *a, int len) {
//   int tid = threadIdx.x;
//   if (tid >= len)
//     return;
//   a[tid] += 'A' - 'a';
// }

__global__ void hello(char *a, int len)
{
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int bz = blockIdx.z;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int bid = (bz * gridDim.y * gridDim.x) + (by * gridDim.x) + bx;
  int tid = (ty * blockDim.x) + tx;

  int id = (bid * blockDim.x * blockDim.y) + tid;

  if (id < len)
    a[id] += 'A' - 'a';
}

int main()
{
  // original string
  char a[N] = "hello@world";
  // length
  int len = strlen(a);
  // pointer to the string on device
  char *ad;
  // pointer to the final string on host
  char *ah;
  // CUDA returned error code
  cudaError_t rc;

  // allocate space for the string on device (GPU) memory
  cudaMalloc((void **)&ad, N);
  cudaMemcpy(ad, a, N, cudaMemcpyHostToDevice);

  // specify grid and block dimensions
  dim3 gridDimensions(2, 2, 2);
  dim3 blockDimensions(2, 4);

  // launch the kernel
  // hello<<<1, N>>>(ad, len);
  hello<<<gridDimensions, blockDimensions>>>(ad, len);
  cudaDeviceSynchronize();

  // for discrete GPUs, get the data from device memory to host memory
  cudaMemcpy(a, ad, N, cudaMemcpyDeviceToHost);
  ah = a;

  // was there any error?
  rc = cudaGetLastError();
  if (rc != cudaSuccess)
    printf("Last CUDA error %s\n", cudaGetErrorString(rc));

  // print final string
  printf("%s!\n", ah);

  // free memory
  cudaFree(ad);

  return 0;
}
