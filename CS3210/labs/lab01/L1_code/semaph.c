/*******************************************************************
 * semaph.c
 * Demonstrates using semaphores to synchromize Linux processes created with
 *fork() Compile: gcc -pthread -o semaph semaph.c Run: ./semaph Partially
 *adapted from
 *https://stackoverflow.com/questions/16400820/c-how-to-use-posix-semaphores-on-forked-processes
 *******************************************************************/

#include <errno.h>
#include <fcntl.h>
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char **argv) {
  /*      loop variables          */
  int i;
  /*      shared memory key       */
  key_t shmkey;
  /*      shared memory id        */
  int shmid;
  /*      synch semaphore         */ /*shared */
  sem_t *sem;
  /*      fork pid                */
  pid_t pid;
  /*      shared variable         */ /*shared */
  int *p;
  /*      fork count              */
  unsigned int n;
  /*      semaphore value         */
  unsigned int value;

  /* initialize a shared variable in shared memory */
  shmkey = ftok("/dev/null", 5); /* valid directory name and a number */
  printf("shmkey for p = %d\n", shmkey);
  shmid = shmget(shmkey, sizeof(int), 0644 | IPC_CREAT);
  if (shmid < 0) { /* shared memory error check */
    perror("shmget\n");
    exit(1);
  }

  p = (int *)shmat(shmid, NULL, 0); /* attach p to shared memory */
  *p = 0;
  printf("p=%d is allocated in shared memory.\n\n", *p);

  /********************************************************/

  printf("How many children do you want to fork?\n");
  printf("Fork count: ");
  scanf("%u", &n);

  printf("What do you want the semaphore's initial value to be?\n");
  printf("Semaphore value: ");
  scanf("%u", &value);

  /* initialize semaphores for shared processes */
  sem = sem_open("pSem", O_CREAT | O_EXCL, 0644, value);
  /* name of semaphore is "pSem", semaphore is reached using this name */

  printf("Semaphore initialized.\n\n");

  /* fork child processes */
  for (i = 0; i < n; i++) {
    pid = fork();
    if (pid < 0) {
      /* check for error      */
      sem_unlink("pSem");
      sem_close(sem);
      /* unlink prevents the semaphore existing forever */
      /* if a crash occurs during the execution         */
      printf("Fork error.\n");
    } else if (pid == 0)
      break; /* child processes */
  }

  /******************************************************/
  /******************   PARENT PROCESS   ****************/
  /******************************************************/
  if (pid != 0) {
    /* wait for all children to exit */
    while (pid = waitpid(-1, NULL, 0)) {
      if (errno == ECHILD)
        break;
    }

    printf("\nParent: All children have exited.\n");

    /* shared memory detach */
    shmdt(p);
    shmctl(shmid, IPC_RMID, 0);

    /* cleanup semaphores */
    sem_unlink("pSem");
    sem_close(sem);
    /* unlink prevents the semaphore existing forever */
    /* if a crash occurs during the execution         */
    exit(0);
  }

  /******************************************************/
  /******************   CHILD PROCESS   *****************/
  /******************************************************/
  else {
    sem_wait(sem); /* P operation */
    printf("  Child(%d) is in critical section.\n", i);
    sleep(1);
    // *p += i % 3; /* increment *p by 0, 1 or 2 based on i */
    *p += i;
    printf("  Child(%d) new value of *p=%d.\n", i, *p);
    sem_post(sem); /* V operation */
    exit(0);
  }
}
