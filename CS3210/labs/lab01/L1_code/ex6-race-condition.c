/*******************************************************************
 * ex6-race-condition.c
 * Demonstrates a race condition.
 * Compile: gcc -pthread ex6-race-condition.c -o ex6-race-condition.out
 * Run: ./ex6-race-condition.out
 *******************************************************************/

#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define ADD_THREADS 10
#define SUB_THREADS 10

#define INITIAL_VALUE 10

int global_counter = INITIAL_VALUE;

pthread_mutex_t lock;
pthread_cond_t global_counter_threshold_cv;
bool can_continue = false;

void *add(void *threadid) {
  long tid = *(long *)threadid;
  pthread_mutex_lock(&lock);
  printf("add thread #%ld acquired mutex lock!\n", tid);

  /* Critical section starts here */
  global_counter++;
  printf("add thread #%ld incremented global_counter!\n", tid);
  printf("add thread #%ld global_counter = %d\n", tid, global_counter);
  if (global_counter == INITIAL_VALUE + ADD_THREADS) {
    printf("Threshold reached!\n");
    can_continue = true;
    pthread_cond_signal(&global_counter_threshold_cv);
    printf("Condition signal sent.\n");
  }
  /* Critical section ends here */

  printf("add thread #%ld released mutex lock!\n", tid);
  pthread_mutex_unlock(&lock);
  pthread_exit(NULL);
}

void *sub(void *threadid) {
  long tid = *(long *)threadid;
  pthread_mutex_lock(&lock);

  /* Critical section starts here */
  while (!can_continue) {
    printf("sub thread #%ld global_counter = %d waiting... \n", tid,
           global_counter);
    pthread_cond_wait(&global_counter_threshold_cv, &lock);
    printf("Condition signal received.");
  }
  printf("sub thread #%ld acquired mutex lock! \n", tid);
  global_counter--;
  printf("sub thread #%ld decremented global_counter! \n", tid);
  printf("sub thread #%ld global_counter = %d\n", tid, global_counter);
  /* Critical section ends here */

  printf("sub thread #%ld released mutex lock! \n", tid);
  pthread_mutex_unlock(&lock);
  pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
  /* Initialize mutex and condition variable objects */
  pthread_mutex_init(&lock, NULL);
  pthread_cond_init(&global_counter_threshold_cv, NULL);

  pthread_t add_threads[ADD_THREADS];
  pthread_t sub_threads[SUB_THREADS];
  long add_threadid[ADD_THREADS];
  long sub_threadid[SUB_THREADS];

  int rc;
  long t1, t2;
  for (t1 = 0; t1 < ADD_THREADS; t1++) {
    int tid = t1;
    add_threadid[tid] = tid;
    printf("main thread: creating add thread %d\n", tid);
    rc = pthread_create(&add_threads[tid], NULL, add,
                        (void *)&add_threadid[tid]);
    if (rc) {
      printf("Return code from pthread_create() is %d\n", rc);
      exit(-1);
    }
  }

  for (t2 = 0; t2 < SUB_THREADS; t2++) {
    int tid = t2;
    sub_threadid[tid] = tid;
    printf("main thread: creating sub thread %d\n", tid);
    rc = pthread_create(&sub_threads[tid], NULL, sub,
                        (void *)&sub_threadid[tid]);
    if (rc) {
      printf("Return code from pthread_create() is %d\n", rc);
      exit(-1);
    }
  }

  /* Join threads */
  for (t1 = 0; t1 < ADD_THREADS; t1++) {
    pthread_join(add_threads[t1], NULL);
  }

  for (t2 = 0; t2 < SUB_THREADS; t2++) {
    pthread_join(sub_threads[t2], NULL);
  }

  printf("### global_counter final value = %d ###\n", global_counter);

  /* Clean up and exit */
  pthread_mutex_destroy(&lock);
  pthread_cond_destroy(&global_counter_threshold_cv);
  pthread_exit(NULL);
}
