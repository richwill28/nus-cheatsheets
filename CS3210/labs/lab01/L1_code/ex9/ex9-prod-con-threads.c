/*******************************************************************************
 * ex9-prod-con-threads.c
 * Producer-consumer synchronization problem in C using threads and condition
 * variables.
 * Compile: gcc -pthread ex9-prod-con-threads.c -o ex9-prod-con-threads.out
 * Run: ./ex9-prod-con-threads.out <BUFFER_CAPACITY> <PRODUCER_LIMIT>
 *      <CONSUMER_LIMIT>
 *
 * Author: Richard Willie (A0219710L)
 ******************************************************************************/

#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define PRODUCERS 2
#define CONSUMERS 1

int BUFFER_CAPACITY;

int *buffer;
int buffer_size = 0;
void push(int item) { buffer[buffer_size++] = item; }
int pop() { return buffer[--buffer_size]; }

int consumer_sum = 0;

int PRODUCER_LIMIT = 1000;
int CONSUMER_LIMIT = 2000;

pthread_mutex_t mutex_buffer;
pthread_cond_t cond_items;
pthread_cond_t cond_spaces;

void *producer(void *threadid) {
  int tid = *(int *)threadid;

  int produced = 0;
  while (produced < PRODUCER_LIMIT) {
    pthread_mutex_lock(&mutex_buffer);
    while (buffer_size == BUFFER_CAPACITY) {
      printf("Producer #%d: Waiting for spaces...\n", tid);
      pthread_cond_wait(&cond_spaces, &mutex_buffer);
    }

    /* Critical section starts here */
    int item = rand() % 11;
    push(item);
    printf("Producer #%d: Add %d to buffer\n", tid, item);
    /* Critical section ends here */

    pthread_mutex_unlock(&mutex_buffer);
    pthread_cond_signal(&cond_items);

    produced++;
  }

  /* Terminate thread */
  pthread_exit(NULL);
}

void *consumer(void *threadid) {
  int tid = *(int *)threadid;

  int consumed = 0;
  while (consumed < CONSUMER_LIMIT) {
    pthread_mutex_lock(&mutex_buffer);
    while (buffer_size == 0) {
      printf("Consumer #%d: Waiting for items...\n", tid);
      pthread_cond_wait(&cond_items, &mutex_buffer);
    }

    /* Critical section starts here */
    int item = pop();
    consumer_sum += item;
    printf("Consumer #%d: Get %d from buffer and add to consumer_sum\n", tid,
           item);
    printf("Consumer #%d: consumer_sum = %d\n", tid, consumer_sum);
    /* Critical section ends here */

    pthread_mutex_unlock(&mutex_buffer);
    pthread_cond_signal(&cond_spaces);

    consumed++;
  }

  /* Terminate thread */
  pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
  if (argc != 2 && argc != 4) {
    printf("Run: ./ex9-prod-con-threads.out <BUFFER_CAPACITY>\n");
    printf("     ./ex9-prod-con-threads.out <BUFFER_CAPACITY> <PRODUCER_LIMIT> "
           "<CONSUMER_LIMIT>\n");
    exit(-1);
  }

  BUFFER_CAPACITY = atoi(argv[1]);
  if (argc == 4) {
    PRODUCER_LIMIT = atoi(argv[2]);
    CONSUMER_LIMIT = atoi(argv[3]);
  }

  /* Allocate memory for buffer */
  buffer = malloc(BUFFER_CAPACITY * sizeof(int));

  pthread_t producer_threads[PRODUCERS];
  pthread_t consumer_threads[CONSUMERS];
  int producer_threadid[PRODUCERS];
  int consumer_threadid[CONSUMERS];

  /* Initialize mutex and condition variables */
  pthread_mutex_init(&mutex_buffer, NULL);
  pthread_cond_init(&cond_items, NULL);
  pthread_cond_init(&cond_spaces, NULL);

  int rc;
  int t1, t2;
  for (t1 = 0; t1 < PRODUCERS; t1++) {
    int tid = t1;
    producer_threadid[tid] = tid;
    printf("Main: creating producer %d\n", tid);
    rc = pthread_create(&producer_threads[tid], NULL, producer,
                        (void *)&producer_threadid[tid]);
    if (rc) {
      printf("Error: Return code from pthread_create() is %d\n", rc);
      exit(-1);
    }
  }

  for (t2 = 0; t2 < CONSUMERS; t2++) {
    int tid = t2;
    consumer_threadid[tid] = tid;
    printf("Main: creating consumer %d\n", tid);
    rc = pthread_create(&consumer_threads[tid], NULL, consumer,
                        (void *)&consumer_threadid[tid]);
    if (rc) {
      printf("Error: Return code from pthread_create() is %d\n", rc);
      exit(-1);
    }
  }

  /* Join threads */
  for (t1 = 0; t1 < PRODUCERS; t1++) {
    pthread_join(producer_threads[t1], NULL);
  }

  for (t2 = 0; t2 < CONSUMERS; t2++) {
    pthread_join(consumer_threads[t2], NULL);
  }

  /* Clean up and exit */
  free(buffer);
  pthread_mutex_destroy(&mutex_buffer);
  pthread_cond_destroy(&cond_items);
  pthread_cond_destroy(&cond_spaces);
  pthread_exit(NULL);
}
