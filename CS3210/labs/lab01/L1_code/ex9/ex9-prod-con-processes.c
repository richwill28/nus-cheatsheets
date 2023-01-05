/*******************************************************************************
 * ex9-prod-con-processes.c
 * Producer-consumer synchronization problem in C using processes and
 * semaphores.
 * Compile: gcc -pthread ex9-prod-con-processes.c -o ex9-prod-con-processes.out
 * Run: ./ex9-prod-con-processes.out <BUFFER_CAPACITY> <PRODUCER_LIMIT>
 *      <CONSUMER_LIMIT>
 *
 * Author: Richard Willie (A0219710L)
 ******************************************************************************/

#include <errno.h>
#include <fcntl.h>
#include <semaphore.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/shm.h>
#include <sys/wait.h>
#include <unistd.h>

#define PRODUCERS 2
#define CONSUMERS 1

int BUFFER_CAPACITY;

int PRODUCER_LIMIT = 1000;
int CONSUMER_LIMIT = 2000;

sem_t *sem_buffer;
sem_t *sem_items;
sem_t *sem_spaces;

void push(int item, int *buffer, int *buffer_size) {
  buffer[(*buffer_size)++] = item;
}

int pop(int *buffer, int *buffer_size) { return buffer[--(*buffer_size)]; }

void producer(int id, int *buffer, int *buffer_size) {
  int produced = 0;
  while (produced < PRODUCER_LIMIT) {
    sem_wait(sem_spaces);
    sem_wait(sem_buffer);

    /* Critical section starts here */
    int item = rand() % 11;
    push(item, buffer, buffer_size);
    printf("Producer #%d: Add %d to buffer\n", id, item);
    /* Critical section ends here */

    sem_post(sem_buffer);
    sem_post(sem_items);

    produced++;
  }
}

void consumer(int id, int *buffer, int *buffer_size, int *consumer_sum) {
  int consumed = 0;
  while (consumed < CONSUMER_LIMIT) {
    sem_wait(sem_items);
    sem_wait(sem_buffer);

    /* Critical section starts here */
    int item = pop(buffer, buffer_size);
    *consumer_sum += item;
    printf("Consumer #%d: Get %d from buffer and add to consumer_sum\n", id,
           item);
    printf("Consumer #%d: consumer_sum = %d\n", id, *consumer_sum);
    /* Critical section ends here */

    sem_post(sem_buffer);
    sem_post(sem_spaces);

    consumed++;
  }
}

int main(int argc, char *argv[]) {
  if (argc != 2 && argc != 4) {
    printf("Run: ./ex9-prod-con-processes.out <BUFFER_CAPACITY>\n");
    printf(
        "     ./ex9-prod-con-processes.out <BUFFER_CAPACITY> <PRODUCER_LIMIT> "
        "<CONSUMER_LIMIT>\n");
    exit(-1);
  }

  BUFFER_CAPACITY = atoi(argv[1]);
  if (argc == 4) {
    PRODUCER_LIMIT = atoi(argv[2]);
    CONSUMER_LIMIT = atoi(argv[3]);
  }

  /* Allocate a shared memory for buffer */
  key_t shmkey_buffer = ftok("/dev/null", 5);
  int shmid_buffer =
      shmget(shmkey_buffer, BUFFER_CAPACITY * sizeof(int), 0644 | IPC_CREAT);
  if (shmid_buffer < 0) {
    perror("shmget\n");
    exit(1);
  }

  /* Allocate a shared memory for buffer size */
  key_t shmkey_buffer_size = ftok("/dev/null", 6);
  int shmid_buffer_size =
      shmget(shmkey_buffer_size, sizeof(int), 0644 | IPC_CREAT);
  if (shmid_buffer_size < 0) {
    perror("shmget\n");
    exit(1);
  }

  /* Allocate a shared memory for consumer sum */
  key_t shmkey_consumer_sum = ftok("/dev/null", 7);
  int shmid_consumer_sum =
      shmget(shmkey_consumer_sum, sizeof(int), 0644 | IPC_CREAT);
  if (shmid_consumer_sum < 0) {
    perror("shmget\n");
    exit(1);
  }

  /* Attach buffer to shared memory */
  int *buffer = (int *)shmat(shmid_buffer, NULL, 0);

  /* Attach buffer size to shared memory */
  int *buffer_size = (int *)shmat(shmid_buffer_size, NULL, 0);

  /* Attach buffer size to shared memory */
  int *consumer_sum = (int *)shmat(shmid_consumer_sum, NULL, 0);

  /* Allocate semaphores */
  sem_buffer = sem_open("buffer", O_CREAT | O_EXCL, 0644, 1);
  sem_items = sem_open("items", O_CREAT | O_EXCL, 0644, 0);
  sem_spaces = sem_open("spaces", O_CREAT | O_EXCL, 0644, BUFFER_CAPACITY);

  /* Fork processes */
  pid_t pid;
  int i;
  for (i = 0; i < PRODUCERS + CONSUMERS; i++) {
    pid = fork();
    if (pid == 0) {
      /* Child process */
      break;
    } else if (pid < 0) {
      /* Handle error */
      sem_unlink("buffer");
      sem_close(sem_buffer);
      sem_unlink("items");
      sem_close(sem_items);
      sem_unlink("spaces");
      sem_close(sem_spaces);
      printf("Fork error.\n");
    }
  }

  if (pid == 0) {
    /* Child process */
    if (i < PRODUCERS) {
      producer(i, buffer, buffer_size);
    } else {
      consumer(i - PRODUCERS, buffer, buffer_size, consumer_sum);
    }
  } else {
    /* Parent process */
    /* Wait for all children to exit */
    while (pid = waitpid(-1, NULL, 0)) {
      if (errno == ECHILD)
        break;
    }
    printf("Parent: All children have exited.\n");

    /* Detach shared memory */
    shmdt(buffer);
    shmctl(shmid_buffer, IPC_RMID, 0);
    shmdt(buffer_size);
    shmctl(shmid_buffer_size, IPC_RMID, 0);
    shmdt(consumer_sum);
    shmctl(shmid_consumer_sum, IPC_RMID, 0);

    /* Clean up semaphores */
    sem_unlink("buffer");
    sem_close(sem_buffer);
    sem_unlink("items");
    sem_close(sem_items);
    sem_unlink("spaces");
    sem_close(sem_spaces);
  }
}
