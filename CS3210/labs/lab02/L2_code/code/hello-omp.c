#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  int no_threads;

/*
 * Fork worker threads, each with its own unique thread id
 *
 * WARNING: any variables declared before this parallel region
 * (e.g. no_threads above) are SHARED amongst all OpenMP threads
 * by default unless specified otherwise with the private clause
 */
#pragma omp parallel
  {
    /* Obtain thread id */
    int thread_id = omp_get_thread_num();
    printf("Hello World from thread = %d\n", thread_id);

    /*
     * Only master thread executes this block
     * Master thread always has thread id of 0
     */
    if (thread_id == 0) {
      no_threads = omp_get_num_threads();
      printf("Number of threads = %d\n", no_threads);
    }
  }
  /* All worker threads join master thread and are destroyed */
}
