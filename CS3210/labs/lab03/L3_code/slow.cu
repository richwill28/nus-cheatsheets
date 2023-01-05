/**
 * Executes useless work on the same number of threads.
 * The only difference is how they are laid out.
 * What to observe/ponder:
 * - The execution time of both layouts
 * - Why is one slower than the other?
 */

#include <stdio.h>

#define HOW_MUCH_WORK 10000

__device__ int result[10];

long long wall_clock_time()
{
#ifdef __linux__
  struct timespec tp;
  clock_gettime(CLOCK_REALTIME, &tp);
  return (long long)(tp.tv_nsec + (long long)tp.tv_sec * 1000000000ll);
#else
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (long long)(tv.tv_usec * 1000 + (long long)tv.tv_sec * 1000000000ll);
#endif
}

void check_cuda_errors()
{
  cudaError_t rc;
  rc = cudaGetLastError();
  if (rc != cudaSuccess)
  {
    fprintf(stderr, "Last CUDA error %s\n", cudaGetErrorString(rc));
  }
}

__global__ void work()
{
  int i;
  for (i = 0; i < HOW_MUCH_WORK; i++)
  {
    result[0]++;
  }
}

int main(int argc, char **argv)
{
  int start, end;

  start = wall_clock_time();
  work<<<1024, 1>>>();
  cudaDeviceSynchronize();
  end = wall_clock_time();
  check_cuda_errors();

  fprintf(stderr, "1024 blocks of 1 thread each took %1.2f seconds\n", ((float)(end - start)) / 1000000000);

  start = wall_clock_time();
  work<<<1, 1024>>>();
  cudaDeviceSynchronize();
  end = wall_clock_time();
  check_cuda_errors();

  fprintf(stderr, "1 block of 1024 threads each took %1.2f seconds\n", ((float)(end - start)) / 1000000000);

  return 0;
}
