## Part 1: Processes vs. Threads

### Exercise 1

For the line to only print once, we can call it before creating the child process.

### Exercise 2

Observation:

- The threads did not execute in the same order they are spawned.
  - Reason: Each thread's routine is executed with varying delay.
- The final value of the variable counter is not always the same.
  - Reason: Many threads are incrementing `counter` at the same time and the operation is not atomic.

## Part 2: Process and Thread Synchronization

Observation for `semaph.c`:

- As expected, when binary semaphore was used, the final value of `*p` is always the same.
- When there are many child processes entering the critical section (i.e., `fork > 1` and `sem > 2`), the final value of `*p` is not consistent.

> Check the differences between named and unnamed semaphore.

### Exercise 3

Observation:

- The final result of the global counter is printed before the completion of all threads.

### Exercise 4

Call `pthread_join` for each thread before final result is printed.

Observation:

- The final result may still be inconsistent due to race condition.

Questions:

- Is it possible that a child thread started and finished before `pthead_join` is invoked in the parent thread?

### Exercise 5

The critical section is where the global counter is modified.

Observation:

- The output is consistent after implementing mutex to the critical section.

> What do you think are the differences between a pthread mutex and a binary semaphore?

### Exercise 6

Readings:

- [Understanding `pthread_cond_wait` and `pthread_cond_signal`](https://stackoverflow.com/questions/16522858/understanding-of-pthread-cond-wait-and-pthread-cond-signal)

Notes:

- Referencing a mutex before initializing it is an [undefined behavior](https://docs.oracle.com/cd/E86824_01/html/E54766/pthread-mutex-init-3c.html).
- Referencing a mutex after it is destroyed (e.g. using `pthread_mutex_destroy`) is also an [undefined behavior](https://docs.oracle.com/cd/E86824_01/html/E54766/pthread-mutex-init-3c.html).
