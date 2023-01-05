## Part 1: Compiling and Running MPI Programs Locally

### Exercise 1

- The ordering of the messages seems random.
- There is a limit on the number of processors.
- Specify the option `--oversubscribe` to use the max number of processors.

### Exercise 2

- The option `--bind-to core` binds each process to each core of the machine.
  - If there are 10 cores on the machine, then we can only launch 10 processes.
- The option `--bind-to hwthread` binds each process to each hardware thread of the machine.
  - If there are 10 cores on the machine and each core has 2 hardware threads, then we can launch 20 processes.

## Part 2: Running MPI Programs across Multiple Nodes

### Exercise 3

- Process 0 ran on pdc-001, while process 1 ran on pdc-002.
- Increasing the number of processes causes an error.

### Exercise 4

```
rank 0=soctf-pdc-001 slot=0:0
rank 1=soctf-pdc-001 slot=0:1
rank 2=soctf-pdc-002 slot=0:0
rank 3=soctf-pdc-002 slot=0:1
```

### Exercise 5

- The maximum number of tasks seems to be limited by the number of logical CPU cores?

## Part 3: Mapping MPI Processes to Nodes / Cores

### Exercise 6

- With `--distribution=block,Pack`, the processes are allocated to the nodes in a greedy manner?
  - For example, node 0 handles 6 processes while node 1 and 2 handle 1 process each.

### Exercise 7

- We can specify arbitrary tasks distribution with `SLURM_HOSTFILE`.
- The number of tasks cannot be increase beyond the number of nodes specified in `SLURM_HOSTFILE`.

### Exercise 8

```shell
$ srun --partition i7-7700 lstopo-no-graphics --taskset

Machine (31GB total) cpuset=0xff
  Package L#0 cpuset=0xff
    NUMANode L#0 (P#0 31GB) cpuset=0xff
    L3 L#0 (8192KB) cpuset=0xff
      L2 L#0 (256KB) cpuset=0x11
        L1d L#0 (32KB) cpuset=0x11
          L1i L#0 (32KB) cpuset=0x11
            Core L#0 cpuset=0x11
              PU L#0 (P#0) cpuset=0x1
              PU L#1 (P#4) cpuset=0x10
      L2 L#1 (256KB) cpuset=0x22
        L1d L#1 (32KB) cpuset=0x22
          L1i L#1 (32KB) cpuset=0x22
            Core L#1 cpuset=0x22
              PU L#2 (P#1) cpuset=0x2
              PU L#3 (P#5) cpuset=0x20
      L2 L#2 (256KB) cpuset=0x44
        L1d L#2 (32KB) cpuset=0x44
          L1i L#2 (32KB) cpuset=0x44
            Core L#2 cpuset=0x44
              PU L#4 (P#2) cpuset=0x4
              PU L#5 (P#6) cpuset=0x40
      L2 L#3 (256KB) cpuset=0x88
        L1d L#3 (32KB) cpuset=0x88
          L1i L#3 (32KB) cpuset=0x88
            Core L#3 cpuset=0x88
              PU L#6 (P#3) cpuset=0x8
              PU L#7 (P#7) cpuset=0x80
  HostBridge
    PCIBridge
      PCI 01:00.0 (VGA)
    PCI 00:02.0 (VGA)
    PCI 00:17.0 (RAID)
      Block(Removable Media Device) "sr0"
      Block(Disk) "sda"
    PCI 00:1f.6 (Ethernet)
      Net "enp0s31f6"
```

```shell
$ srun --partition xs-4114 lstopo-no-graphics --taskset

Machine (31GB total) cpuset=0xfffff
  Package L#0 cpuset=0xfffff
    NUMANode L#0 (P#0 31GB) cpuset=0xfffff
    L3 L#0 (14MB) cpuset=0xfffff
      L2 L#0 (1024KB) cpuset=0x401
        L1d L#0 (32KB) cpuset=0x401
          L1i L#0 (32KB) cpuset=0x401
            Core L#0 cpuset=0x401
              PU L#0 (P#0) cpuset=0x1
              PU L#1 (P#10) cpuset=0x400
      L2 L#1 (1024KB) cpuset=0x802
        L1d L#1 (32KB) cpuset=0x802
          L1i L#1 (32KB) cpuset=0x802
            Core L#1 cpuset=0x802
              PU L#2 (P#1) cpuset=0x2
              PU L#3 (P#11) cpuset=0x800
      L2 L#2 (1024KB) cpuset=0x1004
        L1d L#2 (32KB) cpuset=0x1004
          L1i L#2 (32KB) cpuset=0x1004
            Core L#2 cpuset=0x1004
              PU L#4 (P#2) cpuset=0x4
              PU L#5 (P#12) cpuset=0x1000
      L2 L#3 (1024KB) cpuset=0x2008
        L1d L#3 (32KB) cpuset=0x2008
          L1i L#3 (32KB) cpuset=0x2008
            Core L#3 cpuset=0x2008
              PU L#6 (P#3) cpuset=0x8
              PU L#7 (P#13) cpuset=0x2000
      L2 L#4 (1024KB) cpuset=0x4010
        L1d L#4 (32KB) cpuset=0x4010
          L1i L#4 (32KB) cpuset=0x4010
            Core L#4 cpuset=0x4010
              PU L#8 (P#4) cpuset=0x10
              PU L#9 (P#14) cpuset=0x4000
      L2 L#5 (1024KB) cpuset=0x8020
        L1d L#5 (32KB) cpuset=0x8020
          L1i L#5 (32KB) cpuset=0x8020
            Core L#5 cpuset=0x8020
              PU L#10 (P#5) cpuset=0x20
              PU L#11 (P#15) cpuset=0x8000
      L2 L#6 (1024KB) cpuset=0x10040
        L1d L#6 (32KB) cpuset=0x10040
          L1i L#6 (32KB) cpuset=0x10040
            Core L#6 cpuset=0x10040
              PU L#12 (P#6) cpuset=0x40
              PU L#13 (P#16) cpuset=0x10000
      L2 L#7 (1024KB) cpuset=0x20080
        L1d L#7 (32KB) cpuset=0x20080
          L1i L#7 (32KB) cpuset=0x20080
            Core L#7 cpuset=0x20080
              PU L#14 (P#7) cpuset=0x80
              PU L#15 (P#17) cpuset=0x20000
      L2 L#8 (1024KB) cpuset=0x40100
        L1d L#8 (32KB) cpuset=0x40100
          L1i L#8 (32KB) cpuset=0x40100
            Core L#8 cpuset=0x40100
              PU L#16 (P#8) cpuset=0x100
              PU L#17 (P#18) cpuset=0x40000
      L2 L#9 (1024KB) cpuset=0x80200
        L1d L#9 (32KB) cpuset=0x80200
          L1i L#9 (32KB) cpuset=0x80200
            Core L#9 cpuset=0x80200
              PU L#18 (P#9) cpuset=0x200
              PU L#19 (P#19) cpuset=0x80000
  HostBridge
    PCI 00:16.2 (IDE)
    PCI 00:17.0 (RAID)
      Block(Removable Media Device) "sr0"
      Block(Disk) "sda"
    PCI 00:1f.6 (Ethernet)
      Net "enp0s31f6"
  HostBridge
    PCI 16:05.5 (RAID)
  HostBridge
    PCIBridge
      PCI b3:00.0 (VGA)
```

```shell
$ srun -n 4 -p i7-7700 --cpu-bind=verbose,threads /nfs/home/$USER/hello

cpu-bind=MASK - soctf-pdc-013, task  0  0 [72417]: mask 0x1 set
cpu-bind=MASK - soctf-pdc-013, task  2  2 [72419]: mask 0x2 set
cpu-bind=MASK - soctf-pdc-013, task  3  3 [72420]: mask 0x20 set
cpu-bind=MASK - soctf-pdc-013, task  1  1 [72418]: mask 0x10 set
Hello world from process 0 out of 4 on host soctf-pdc-013
Hello world from process 1 out of 4 on host soctf-pdc-013
Hello world from process 2 out of 4 on host soctf-pdc-013
Hello world from process 3 out of 4 on host soctf-pdc-013
```

```shell
$ srun -n 4 -p i7-7700 --cpu-bind=verbose,cores /nfs/home/$USER/hello

cpu-bind=MASK - soctf-pdc-013, task  2  2 [72845]: mask 0x44 set
cpu-bind=MASK - soctf-pdc-013, task  0  0 [72843]: mask 0x11 set
cpu-bind=MASK - soctf-pdc-013, task  3  3 [72846]: mask 0x88 set
cpu-bind=MASK - soctf-pdc-013, task  1  1 [72844]: mask 0x22 set
Hello world from process 2 out of 4 on host soctf-pdc-013
Hello world from process 0 out of 4 on host soctf-pdc-013
Hello world from process 1 out of 4 on host soctf-pdc-013
Hello world from process 3 out of 4 on host soctf-pdc-013
```

## Part 4 Part 4: Process-to-process Communication

### Exercise 9

```shell
$ srun -N 2 /nfs/home/$USER/block_comm

Task 0 on soctf-pdc-005: Received 1 char(s) from task 1 with tag 1 
Task 1 on soctf-pdc-006: Received 1 char(s) from task 0 with tag 1
```

### Exercise 10

Refer to `block_comm_1.c`.

### Exercise 11

- Deadlock occurs because the send and receive operations are blocking.

### Exercise 12

- No deadlock because the send and receive operations are non-blocking.

## Part 5: For Submission

### Exercise 13

```shell
$ srun -n5 -p i7-7700 /nfs/home/$USER/mm-mpi 2048

 --- Process 1 (on soctf-pdc-013) is worker
Usage: /nfs/home/richwill/mm-mpi <size>
 +++ Process 4 (on soctf-pdc-013) is master
Multiplying matrix size 2048 x 2048
 --- Process 0 (on soctf-pdc-013) is worker
 --- Process 2 (on soctf-pdc-013) is worker
 --- Process 3 (on soctf-pdc-013) is worker
 +++ MASTER : Distributing matrix A to workers: workers 4, rows_per_worker 512 
 --- WORKER 0: Received row [0-511] of matrix A
 +++ MASTER : Finished sending row [0-511] of matrix A to process 0
 --- WORKER 0: Receiving all 2048 rows for matrix B...
 --- WORKER 1: Received row [512-1023] of matrix A
 +++ MASTER : Finished sending row [512-1023] of matrix A to process 1
 --- WORKER 1: Receiving all 2048 rows for matrix B...
 --- WORKER 2: Received row [1024-1535] of matrix A
 +++ MASTER : Finished sending row [1024-1535] of matrix A to process 2
 --- WORKER 2: Receiving all 2048 rows for matrix B...
 --- WORKER 3: Received row [1536-2047] of matrix A
 +++ MASTER : Finished sending row [1536-2047] of matrix A to process 3
 +++ MASTER : Sending matrix B to all workers
 --- WORKER 3: Receiving all 2048 rows for matrix B...
 --- WORKER 0: Received matrix B
 --- WORKER 1: Received matrix B
 --- WORKER 2: Received matrix B
 --- WORKER 3: Received matrix B
 +++ MASTER : Finished sending matrix B to all workers
 +++ MASTER: Receiving the results from workers
 --- WORKER 0: Finished the computations
 --- WORKER 0: Sent the results back
 --- WORKER 0 (on soctf-pdc-013): communication_time=  0.21 seconds; computation_time= 18.09 seconds
 --- WORKER 3: Finished the computations
 --- WORKER 2: Finished the computations
 --- WORKER 1: Finished the computations
 --- WORKER 1: Sent the results back
 --- WORKER 1 (on soctf-pdc-013): communication_time=  0.21 seconds; computation_time= 26.40 seconds
 --- WORKER 2: Sent the results back
 --- WORKER 2 (on soctf-pdc-013): communication_time=  0.25 seconds; computation_time= 26.35 seconds
 --- WORKER 3: Sent the results back
 --- WORKER 3 (on soctf-pdc-013): communication_time=  8.45 seconds; computation_time= 18.16 seconds
 --- MASTER 4 (on soctf-pdc-013): total time= 26.61 seconds
```

```shell
$ srun -n5 -p xs-4114 /nfs/home/$USER/mm-mpi 2048

 --- Process 3 (on soctf-pdc-008) is worker
 --- Process 0 (on soctf-pdc-008) is worker
 --- Process 1 (on soctf-pdc-008) is worker
Usage: /nfs/home/richwill/mm-mpi <size>
 +++ Process 4 (on soctf-pdc-008) is master
Multiplying matrix size 2048 x 2048
 --- Process 2 (on soctf-pdc-008) is worker
 +++ MASTER : Distributing matrix A to workers: workers 4, rows_per_worker 512 
 --- WORKER 0: Received row [0-511] of matrix A
 +++ MASTER : Finished sending row [0-511] of matrix A to process 0
 --- WORKER 0: Receiving all 2048 rows for matrix B...
 --- WORKER 1: Received row [512-1023] of matrix A
 +++ MASTER : Finished sending row [512-1023] of matrix A to process 1
 --- WORKER 1: Receiving all 2048 rows for matrix B...
 --- WORKER 2: Received row [1024-1535] of matrix A
 +++ MASTER : Finished sending row [1024-1535] of matrix A to process 2
 --- WORKER 2: Receiving all 2048 rows for matrix B...
 --- WORKER 3: Received row [1536-2047] of matrix A
 +++ MASTER : Finished sending row [1536-2047] of matrix A to process 3
 +++ MASTER : Sending matrix B to all workers
 --- WORKER 3: Receiving all 2048 rows for matrix B...
 --- WORKER 0: Received matrix B
 --- WORKER 1: Received matrix B
 --- WORKER 2: Received matrix B
 --- WORKER 3: Received matrix B
 +++ MASTER : Finished sending matrix B to all workers
 +++ MASTER: Receiving the results from workers
 --- WORKER 2: Finished the computations
 --- WORKER 1: Finished the computations
 --- WORKER 0: Finished the computations
 --- WORKER 0: Sent the results back
 --- WORKER 0 (on soctf-pdc-008): communication_time=  0.31 seconds; computation_time= 28.90 seconds
 --- WORKER 1: Sent the results back
 --- WORKER 1 (on soctf-pdc-008): communication_time=  0.76 seconds; computation_time= 28.45 seconds
 --- WORKER 2: Sent the results back
 --- WORKER 2 (on soctf-pdc-008): communication_time=  1.04 seconds; computation_time= 28.17 seconds
 --- WORKER 3: Finished the computations
 --- WORKER 3: Sent the results back
 --- WORKER 3 (on soctf-pdc-008): communication_time=  0.31 seconds; computation_time= 29.11 seconds
 --- MASTER 4 (on soctf-pdc-008): total time= 29.42 seconds
```

```shell
$ srun -n 5 --nodes 2 --constraint="i7-7700*1 xs-4114*1" /nfs/home/$USER/mm-mpi 2048

Usage: /nfs/home/richwill/mm-mpi <size>
 +++ Process 4 (on soctf-pdc-013) is master
Multiplying matrix size 2048 x 2048
 --- Process 0 (on soctf-pdc-008) is worker
 --- Process 3 (on soctf-pdc-013) is worker
 --- Process 2 (on soctf-pdc-008) is worker
 --- Process 1 (on soctf-pdc-008) is worker
 +++ MASTER : Distributing matrix A to workers: workers 4, rows_per_worker 512 
 +++ MASTER : Finished sending row [0-511] of matrix A to process 0
 --- WORKER 0: Received row [0-511] of matrix A
 --- WORKER 0: Receiving all 2048 rows for matrix B...
 +++ MASTER : Finished sending row [512-1023] of matrix A to process 1
 --- WORKER 1: Received row [512-1023] of matrix A
 --- WORKER 1: Receiving all 2048 rows for matrix B...
 +++ MASTER : Finished sending row [1024-1535] of matrix A to process 2
 --- WORKER 3: Received row [1536-2047] of matrix A
 +++ MASTER : Finished sending row [1536-2047] of matrix A to process 3
 +++ MASTER : Sending matrix B to all workers
 --- WORKER 3: Receiving all 2048 rows for matrix B...
 --- WORKER 2: Received row [1024-1535] of matrix A
 --- WORKER 2: Receiving all 2048 rows for matrix B...
 --- WORKER 3: Received matrix B
 +++ MASTER : Finished sending matrix B to all workers
 +++ MASTER: Receiving the results from workers
 --- WORKER 0: Received matrix B
 --- WORKER 1: Received matrix B
 --- WORKER 2: Received matrix B
 --- WORKER 3: Finished the computations
 --- WORKER 1: Finished the computations
 --- WORKER 1: Sent the results back
 --- WORKER 1 (on soctf-pdc-008): communication_time=  0.79 seconds; computation_time= 27.08 seconds
 --- WORKER 0: Finished the computations
 --- WORKER 2: Finished the computations
 --- WORKER 0: Sent the results back
 --- WORKER 0 (on soctf-pdc-008): communication_time=  0.79 seconds; computation_time= 27.34 seconds
 --- WORKER 2: Sent the results back
 --- WORKER 2 (on soctf-pdc-008): communication_time=  0.81 seconds; computation_time= 27.35 seconds
 --- WORKER 3: Sent the results back
 --- WORKER 3 (on soctf-pdc-013): communication_time= 10.51 seconds; computation_time= 17.67 seconds
 --- MASTER 4 (on soctf-pdc-013): total time= 28.17 seconds
```

### Exercise 14

```shell
$ srun -n5 -N1 -p i7-7700 /nfs/home/$USER/mm-mpi 256

Usage: /nfs/home/richwill/mm-mpi <size>
 +++ Process 4 (on soctf-pdc-013) is master
Multiplying matrix size 256 x 256
 --- Process 3 (on soctf-pdc-013) is worker
 --- Process 2 (on soctf-pdc-013) is worker
 --- Process 0 (on soctf-pdc-013) is worker
 --- Process 1 (on soctf-pdc-013) is worker
 +++ MASTER : Distributing matrix A to workers: workers 4, rows_per_worker 64 
 --- WORKER 0: Received row [0-63] of matrix A
 +++ MASTER : Finished sending row [0-63] of matrix A to process 0
 --- WORKER 0: Receiving all 256 rows for matrix B...
 +++ MASTER : Finished sending row [64-127] of matrix A to process 1
 --- WORKER 1: Received row [64-127] of matrix A
 --- WORKER 2: Received row [128-191] of matrix A
 +++ MASTER : Finished sending row [128-191] of matrix A to process 2
 --- WORKER 1: Receiving all 256 rows for matrix B...
 --- WORKER 2: Receiving all 256 rows for matrix B...
 --- WORKER 3: Received row [192-255] of matrix A
 +++ MASTER : Finished sending row [192-255] of matrix A to process 3
 +++ MASTER : Sending matrix B to all workers
 --- WORKER 3: Receiving all 256 rows for matrix B...
 --- WORKER 0: Received matrix B
 --- WORKER 1: Received matrix B
 --- WORKER 2: Received matrix B
 --- WORKER 3: Received matrix B
 +++ MASTER : Finished sending matrix B to all workers
 +++ MASTER: Receiving the results from workers
 --- WORKER 1: Finished the computations
 --- WORKER 1: Sent the results back
 --- WORKER 1 (on soctf-pdc-013): communication_time=  0.01 seconds; computation_time=  0.03 seconds
 --- WORKER 3: Finished the computations
 --- WORKER 2: Finished the computations
 --- WORKER 3: Sent the results back
 --- WORKER 3 (on soctf-pdc-013): communication_time=  0.01 seconds; computation_time=  0.04 seconds
 --- WORKER 2: Sent the results back
 --- WORKER 2 (on soctf-pdc-013): communication_time=  0.01 seconds; computation_time=  0.04 seconds
 --- WORKER 0: Finished the computations
 --- WORKER 0: Sent the results back
 --- WORKER 0 (on soctf-pdc-013): communication_time=  0.01 seconds; computation_time=  0.04 seconds
 --- MASTER 4 (on soctf-pdc-013): total time=  0.05 seconds
```

```shell
$ srun -n9 -N1 --overcommit -p i7-7700 /nfs/home/$USER/mm-mpi 256

 --- Process 0 (on soctf-pdc-013) is worker
 --- Process 7 (on soctf-pdc-013) is worker
 --- Process 4 (on soctf-pdc-013) is worker
 --- Process 5 (on soctf-pdc-013) is worker
 --- Process 1 (on soctf-pdc-013) is worker
 --- Process 2 (on soctf-pdc-013) is worker
 --- Process 6 (on soctf-pdc-013) is worker
 --- Process 3 (on soctf-pdc-013) is worker
Usage: /nfs/home/richwill/mm-mpi <size>
 +++ Process 8 (on soctf-pdc-013) is master
Multiplying matrix size 256 x 256
 +++ MASTER : Distributing matrix A to workers: workers 8, rows_per_worker 32 
 --- WORKER 0: Received row [0-31] of matrix A
 +++ MASTER : Finished sending row [0-31] of matrix A to process 0
 --- WORKER 0: Receiving all 256 rows for matrix B...
 --- WORKER 1: Received row [32-63] of matrix A
 +++ MASTER : Finished sending row [32-63] of matrix A to process 1
 --- WORKER 2: Received row [64-95] of matrix A
 +++ MASTER : Finished sending row [64-95] of matrix A to process 2
 --- WORKER 1: Receiving all 256 rows for matrix B...
 --- WORKER 2: Receiving all 256 rows for matrix B...
 +++ MASTER : Finished sending row [96-127] of matrix A to process 3
 --- WORKER 3: Received row [96-127] of matrix A
 --- WORKER 4: Received row [128-159] of matrix A
 +++ MASTER : Finished sending row [128-159] of matrix A to process 4
 --- WORKER 3: Receiving all 256 rows for matrix B...
 --- WORKER 5: Received row [160-191] of matrix A
 +++ MASTER : Finished sending row [160-191] of matrix A to process 5
 --- WORKER 4: Receiving all 256 rows for matrix B...
 --- WORKER 5: Receiving all 256 rows for matrix B...
 --- WORKER 6: Received row [192-223] of matrix A
 +++ MASTER : Finished sending row [192-223] of matrix A to process 6
 --- WORKER 6: Receiving all 256 rows for matrix B...
 --- WORKER 7: Received row [224-255] of matrix A
 +++ MASTER : Finished sending row [224-255] of matrix A to process 7
 +++ MASTER : Sending matrix B to all workers
 --- WORKER 7: Receiving all 256 rows for matrix B...
 --- WORKER 0: Received matrix B
 --- WORKER 1: Received matrix B
 --- WORKER 2: Received matrix B
 --- WORKER 3: Received matrix B
 --- WORKER 0: Finished the computations
 --- WORKER 0: Sent the results back
 --- WORKER 0 (on soctf-pdc-013): communication_time=  7.95 seconds; computation_time=  0.01 seconds
 --- WORKER 1: Finished the computations
 --- WORKER 1: Sent the results back
 --- WORKER 1 (on soctf-pdc-013): communication_time=  7.95 seconds; computation_time=  0.02 seconds
 --- WORKER 4: Received matrix B
 --- WORKER 5: Received matrix B
 --- WORKER 6: Received matrix B
 --- WORKER 7: Received matrix B
 +++ MASTER : Finished sending matrix B to all workers
 +++ MASTER: Receiving the results from workers
 --- WORKER 2: Finished the computations
 --- WORKER 2: Sent the results back
 --- WORKER 2 (on soctf-pdc-013): communication_time=  7.95 seconds; computation_time=  0.02 seconds
 --- WORKER 3: Finished the computations
 --- WORKER 3: Sent the results back
 --- WORKER 3 (on soctf-pdc-013): communication_time=  7.95 seconds; computation_time=  0.02 seconds
 --- WORKER 5: Finished the computations
 --- WORKER 5: Sent the results back
 --- WORKER 5 (on soctf-pdc-013): communication_time=  7.96 seconds; computation_time=  0.01 seconds
 --- WORKER 4: Finished the computations
 --- WORKER 4: Sent the results back
 --- WORKER 4 (on soctf-pdc-013): communication_time=  7.96 seconds; computation_time=  0.01 seconds
 --- WORKER 7: Finished the computations
 --- WORKER 6: Finished the computations
 --- WORKER 7: Sent the results back
 --- WORKER 7 (on soctf-pdc-013): communication_time=  7.96 seconds; computation_time=  0.02 seconds
 --- WORKER 6: Sent the results back
 --- WORKER 6 (on soctf-pdc-013): communication_time=  7.96 seconds; computation_time=  0.02 seconds
 --- MASTER 8 (on soctf-pdc-013): total time=  7.96 seconds
```
