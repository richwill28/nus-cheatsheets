#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define ADD_THREADS 10
#define SUB_THREADS 10

// How many adds/subs each thread should do before exiting
#define OPERATIONS 100

// Delay to simulate each add/sub as an equivalent CPU-heavy calculation
#define DELAY 1000000

int global_counter;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

void work_delay() {
  for (int i = 0; i < DELAY; i++) {
  }
}

void *add(void *threadid) {
  long tid = *(long *)threadid;

  for (int i = 0; i < OPERATIONS; i++) {
    pthread_mutex_lock(&lock);
    work_delay();
    global_counter++;
    printf("add thread #%ld incremented global_counter!\n", tid);
    pthread_mutex_unlock(&lock);
    usleep(1);
  }
}

void *sub(void *threadid) {
  long tid = *(long *)threadid;

  for (int i = 0; i < OPERATIONS; i++) {
    pthread_mutex_lock(&lock);
    work_delay();
    global_counter--;
    printf("sub thread #%ld decremented global_counter! \n", tid);
    pthread_mutex_unlock(&lock);
    usleep(1);
  }
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

  // Wait on all ADD and SUB threads to terminate before printing counter
  for (t1 = 0; t1 < ADD_THREADS; t1++) {
    pthread_join(add_threads[t1], NULL);
  }

  for (t2 = 0; t2 < SUB_THREADS; t2++) {
    pthread_join(sub_threads[t2], NULL);
  }

  printf("### global_counter final value = %d ###\n", global_counter);

  // Clean up and exit
  pthread_mutex_destroy(&lock);
  pthread_exit(NULL);
}
