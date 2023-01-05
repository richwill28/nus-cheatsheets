/*******************************************************************
 * processes.c
 * Demonstrates process creation in Linux.
 * Compile: gcc -o processes processes.c
 * Run: ./processes
 ******************************************************************/

#include <stdio.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

int value = 10;

int main(int argc, char *argv[]) {
  pid_t fork_ret;

  printf("We just cloned a process..!\n");

  // create the child process
  fork_ret = fork();
  // printf("We just cloned a process..!\n");

  if (fork_ret < 0) {
    // fork_ret would be -1 if unsuccessful
    fprintf(stderr, "Fork failed!\n");

    return 1;
  } else if (fork_ret == 0) {
    // fork_ret would return 0 in child
    value += 10;
    printf("Child process: value = %d\n", value);
  } else {
    // fork_ret is the PID of the child (in the parent)
    // parent waits until child is completed
    wait(NULL);
    printf("Child completed ....\n");
    printf("Parent process: value = %d\n", value);
  }

  return 0;
}
