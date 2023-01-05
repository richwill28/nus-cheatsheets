/*******************************************************************
 * ex5-race-condition.c
 * Demonstrates a race condition.
 * Compile: gcc -pthread ex5-race-condition.c -o ex5-race-condition.out
 * Run: ./ex5-race-condition.out
 *******************************************************************/

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define ADD_THREADS 1000
#define SUB_THREADS 1000

int global_counter;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

void *add(void *threadid) {
  long tid = *(long *)threadid;
  pthread_mutex_lock(&lock);
  printf("add thread #%ld acquired mutex lock!\n", tid);

  /* Critical section starts here */
  global_counter++;
  printf("add thread #%ld incremented global_counter!\n", tid);
  /* Critical section ends here */

  printf("add thread #%ld released mutex lock!\n", tid);
  pthread_mutex_unlock(&lock);
}

void *sub(void *threadid) {
  long tid = *(long *)threadid;
  pthread_mutex_lock(&lock);
  printf("sub thread #%ld acquired mutex lock! \n", tid);

  /* Critical section starts here */
  global_counter--;
  printf("sub thread #%ld decremented global_counter! \n", tid);
  /* Critical section ends here */

  printf("sub thread #%ld released mutex lock! \n", tid);
  pthread_mutex_unlock(&lock);
}

int main(int argc, char *argv[]) {
  global_counter = 10;
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
  pthread_exit(NULL);
}
