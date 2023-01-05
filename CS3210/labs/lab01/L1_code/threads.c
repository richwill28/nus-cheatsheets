/*******************************************************************
 * threads.c
 * Demonstrates thread creation and termination.
 * Compile: gcc -pthread -o threads threads.c
 * Run: ./threads
 ******************************************************************/

#include <pthread.h> // include the pthread library
#include <stdio.h>
#include <stdlib.h>

#define NUM_THREADS 1000

int counter = 0;

// function to run in parallel
void *work(void *threadid) {
  long tid = *(long *)threadid;
  counter++;
  printf("thread #%ld incrementing counter. counter = %d\n", tid, counter);
  pthread_exit(NULL); // terminate thread
}

int main(int argc, char *argv[]) {
  pthread_t threads[NUM_THREADS]; // reference to threads
  long threadids[NUM_THREADS];
  int rc;
  long t;
  for (t = 0; t < NUM_THREADS; t++) {
    printf("main thread: creating thread %ld\n", t);
    threadids[t] = t;

    // pthread_create spawns a new thread and return 0 on success
    rc = pthread_create(&threads[t], NULL, work, (void *)&threadids[t]);
    if (rc) {
      printf("Return code from pthread_create() is %d\n", rc);
      exit(-1);
    }
  }
  pthread_exit(NULL);
}
