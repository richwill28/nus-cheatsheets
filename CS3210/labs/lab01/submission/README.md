## Exercise 7

Compile:

```shell
gcc -pthread ex7-prod-con-threads.c -o ex7-prod-con-threads.out
```

Run:

```shell
./ex7-prod-con-threads.out <THREAD_DELAY> <BUFFER_CAPACITY>
```

Notes:

- `THREAD_DELAY`: The delay (in second) before the each thread execute its routine.
- `BUFFER_CAPACITY`: The capacity of the buffer, i.e. how many items can the buffer holds.

## Exercise 8

Compile:

```shell
gcc -pthread ex8-prod-con-processes.c -o ex8-prod-con-processes.out
```

Run:

```shell
./ex8-prod-con-processes.out <PROCESS_DELAY> <BUFFER_CAPACITY>
```

Notes:

- `PROCESS_DELAY`: The delay (in second) before the each process execute its routine.
- `BUFFER_CAPACITY`: The capacity of the buffer, i.e. how many items can the buffer holds.

## Exercise 9

Compile:

```shell
gcc -pthread ex9-prod-con-threads.c -o ex9-prod-con-threads.out

gcc -pthread ex9-prod-con-processes.c -o ex9-prod-con-processes.out
```

Run:

```shell
./ex9-prod-con-threads.out <BUFFER_CAPACITY>
./ex9-prod-con-threads.out <BUFFER_CAPACITY> <PRODUCER_LIMIT> <CONSUMER_LIMIT>

./ex9-prod-con-processes.out <BUFFER_CAPACITY>
./ex9-prod-con-processes.out <BUFFER_CAPACITY> <PRODUCER_LIMIT> <CONSUMER_LIMIT>
```

Notes:

- `BUFFER_CAPACITY`: The capacity of the buffer, i.e. how many items can the buffer hold.
- `PRODUCER_LIMIT`: The number of items produced by each producer. (Default is 1000)
- `CONSUMER_LIMIT`: The number of items consumed by the consumer. (Default is 2000)

Observation:

- Running each code empirically, it is observed that the process implementation is faster than the thread implementation. This suggests that performance gain is more significant when parallelism is achieved through multiprocessing, as compared to the context-switching in multithreading.
- When checking each program with `htop`, it is observed that the process implementation has higher CPU utilization than the thread implementation. My hyppthesis is as follows:
  - The `pthread` library creates user-level threads. They rely on the OS scheduler which is not context aware, thus the threads have lower priority and are not able to execute on time, and/or fully utilize the available resources.
  - On the other hand, the `fork` implementation allows the OS to be aware of the processes running. The OS can thus handle resources allocation better with multiple processors.

## Known issues

- When running `ex8-prod-con-processes`, all the child processes may occasionally terminate immediately, while I couldn't figure out why.
