## Part 1: Shared-Memory Multi-Threaded Programming with OpenMP

### Exercise 1

Observation:

```shell
$ /usr/bin/time -vvv ./mm0.out 500
Sequential matrix multiplication of size 500
Matrix multiplication took 0.51 seconds
        Command being timed: "./mm0.out 500"
        User time (seconds): 0.52
        System time (seconds): 0.00
        Percent of CPU this job got: 100%
        Elapsed (wall clock) time (h:mm:ss or m:ss): 0:00.52
        Average shared text size (kbytes): 0
        Average unshared data size (kbytes): 0
        Average stack size (kbytes): 0
        Average total size (kbytes): 0
        Maximum resident set size (kbytes): 4304
        Average resident set size (kbytes): 0
        Major (requiring I/O) page faults: 0
        Minor (reclaiming a frame) page faults: 807
        Voluntary context switches: 1
        Involuntary context switches: 1
        Swaps: 0
        File system inputs: 0
        File system outputs: 0
        Socket messages sent: 0
        Socket messages received: 0
        Signals delivered: 0
        Page size (bytes): 4096
        Exit status: 0

$ /usr/bin/time -vvv ./mm0.out 1000
Sequential matrix multiplication of size 1000
Matrix multiplication took 5.16 seconds
        Command being timed: "./mm0.out 1000"
        User time (seconds): 5.19
        System time (seconds): 0.00
        Percent of CPU this job got: 99%
        Elapsed (wall clock) time (h:mm:ss or m:ss): 0:05.20
        Average shared text size (kbytes): 0
        Average unshared data size (kbytes): 0
        Average stack size (kbytes): 0
        Average total size (kbytes): 0
        Maximum resident set size (kbytes): 13284
        Average resident set size (kbytes): 0
        Major (requiring I/O) page faults: 0
        Minor (reclaiming a frame) page faults: 3013
        Voluntary context switches: 1
        Involuntary context switches: 1
        Swaps: 0
        File system inputs: 0
        File system outputs: 0
        Socket messages sent: 0
        Socket messages received: 0
        Signals delivered: 0
        Page size (bytes): 4096
        Exit status: 0
```

The matrix multiplication took 0.51 and 5.16 seconds for n = 500 and n = 1000, respectively. In other words, the running time increased by a factor of 10 when the input size was doubled. This result is expected because the time complexity of the algorithm is O(n^3).

### Exercise 2

## Part 2: Performance Instrumentation

Observation:

```shell
$ ./mm1.out 2048 8
Matrix multiplication of size 2048 using 8 threads
Matrix multiplication took 15.93 seconds

$ ./mm1.out 2048 16
Matrix multiplication of size 2048 using 16 threads
Matrix multiplication took 11.66 seconds

$ ./mm1.out 2048 32
Matrix multiplication of size 2048 using 32 threads
Matrix multiplication took 9.41 seconds

$ ./mm1.out 2048 64
Matrix multiplication of size 2048 using 64 threads
Matrix multiplication took 9.17 seconds

$ ./mm1.out 2048 128
Matrix multiplication of size 2048 using 128 threads
Matrix multiplication took 9.06 seconds

$ ./mm1.out 2048 256
Matrix multiplication of size 2048 using 256 threads
Matrix multiplication took 8.88 seconds

$ ./mm1.out 2048 512
Matrix multiplication of size 2048 using 512 threads
Matrix multiplication took 8.76 seconds

$ ./mm1.out 2048 1024
Matrix multiplication of size 2048 using 1024 threads
Matrix multiplication took 8.69 seconds

$ ./mm1.out 2048 2048
Matrix multiplication of size 2048 using 2048 threads
Matrix multiplication took 8.63 seconds
```

The matrix multiplication took less time to run as the number of threads increased. However, notice that the reduction in time is getting smaller. Why is this happening?

Hypothesis: the overhead of managing the threads is becoming more significant than the cost of the computation. 

What determines the ideal number of threads to use?

> TODO: Answer

### Exercise 3

Note: We can supply option `-r` for `perf` to make it run many times.

Observation:

```shell
$ perf stat -- ./mm1.out 1024 8
Matrix multiplication of size 1024 using 8 threads
Matrix multiplication took 1.38 seconds

 Performance counter stats for './mm1.out 1024 8':

          11325.10 msec task-clock                #    7.806 CPUs utilized
                50      context-switches          #    0.004 K/sec
                 0      cpu-migrations            #    0.000 K/sec
              3191      page-faults               #    0.282 K/sec
       28802058668      cycles                    #    2.543 GHz
       63617541671      instructions              #    2.21  insn per cycle
        2201578805      branches                  #  194.398 M/sec
           1224213      branch-misses             #    0.06% of all branches

       1.450906646 seconds time elapsed

      11.310852000 seconds user
       0.016009000 seconds sys

$ perf stat -- ./mm1.out 1024 16
Matrix multiplication of size 1024 using 16 threads
Matrix multiplication took 1.27 seconds

 Performance counter stats for './mm1.out 1024 16':

          17829.21 msec task-clock                #   13.307 CPUs utilized
                82      context-switches          #    0.005 K/sec
                13      cpu-migrations            #    0.001 K/sec
              3206      page-faults               #    0.180 K/sec
       43546526171      cycles                    #    2.442 GHz
       63663368493      instructions              #    1.46  insn per cycle
        2213887177      branches                  #  124.172 M/sec
           1286472      branch-misses             #    0.06% of all branches

       1.339878649 seconds time elapsed

      17.817602000 seconds user
       0.016005000 seconds sys

$ perf stat -- ./mm1.out 1024 32
Matrix multiplication of size 1024 using 32 threads
Matrix multiplication took 1.18 seconds

 Performance counter stats for './mm1.out 1024 32':

          21329.20 msec task-clock                #   16.979 CPUs utilized
              3951      context-switches          #    0.185 K/sec
               101      cpu-migrations            #    0.005 K/sec
              3241      page-faults               #    0.152 K/sec
       52396592917      cycles                    #    2.457 GHz
       63642904462      instructions              #    1.21  insn per cycle
        2207682517      branches                  #  103.505 M/sec
           1557108      branch-misses             #    0.07% of all branches

       1.256232439 seconds time elapsed

      21.287280000 seconds user
       0.056050000 seconds sys

$ perf stat -- ./mm1.out 1024 64
Matrix multiplication of size 1024 using 64 threads
Matrix multiplication took 1.08 seconds

 Performance counter stats for './mm1.out 1024 64':

          21304.30 msec task-clock                #   18.427 CPUs utilized
              2482      context-switches          #    0.117 K/sec
               253      cpu-migrations            #    0.012 K/sec
              3311      page-faults               #    0.155 K/sec
       52427545492      cycles                    #    2.461 GHz
       63619087552      instructions              #    1.21  insn per cycle
        2199692630      branches                  #  103.251 M/sec
           1374435      branch-misses             #    0.06% of all branches

       1.156121222 seconds time elapsed

      21.274268000 seconds user
       0.036044000 seconds sys

$ perf stat -- ./mm1.out 1024 128
Matrix multiplication of size 1024 using 128 threads
Matrix multiplication took 1.09 seconds

 Performance counter stats for './mm1.out 1024 128':

          21389.47 msec task-clock                #   18.352 CPUs utilized
              5570      context-switches          #    0.260 K/sec
               429      cpu-migrations            #    0.020 K/sec
              3448      page-faults               #    0.161 K/sec
       52565776343      cycles                    #    2.458 GHz
       63641579848      instructions              #    1.21  insn per cycle
        2204359683      branches                  #  103.058 M/sec
           1541461      branch-misses             #    0.07% of all branches

       1.165484448 seconds time elapsed

      21.309675000 seconds user
       0.095612000 seconds sys

$ perf stat -- ./mm1.out 1024 256
Matrix multiplication of size 1024 using 256 threads
Matrix multiplication took 1.08 seconds

 Performance counter stats for './mm1.out 1024 256':

          21509.68 msec task-clock                #   18.437 CPUs utilized
              6360      context-switches          #    0.296 K/sec
               682      cpu-migrations            #    0.032 K/sec
              3722      page-faults               #    0.173 K/sec
       52767512195      cycles                    #    2.453 GHz
       63648898125      instructions              #    1.21  insn per cycle
        2205962370      branches                  #  102.557 M/sec
           1479197      branch-misses             #    0.07% of all branches

       1.166682288 seconds time elapsed

      21.483033000 seconds user
       0.040102000 seconds sys

$ perf stat -- ./mm1.out 1024 512
Matrix multiplication of size 1024 using 512 threads
Matrix multiplication took 1.07 seconds

 Performance counter stats for './mm1.out 1024 512':

          21271.71 msec task-clock                #   18.193 CPUs utilized
              8218      context-switches          #    0.386 K/sec
              1772      cpu-migrations            #    0.083 K/sec
              4274      page-faults               #    0.201 K/sec
       52123156706      cycles                    #    2.450 GHz
       63687045476      instructions              #    1.22  insn per cycle
        2213889134      branches                  #  104.077 M/sec
           1655585      branch-misses             #    0.07% of all branches

       1.169219674 seconds time elapsed

      21.118334000 seconds user
       0.183811000 seconds sys

$ perf stat -- ./mm1.out 1024 1024
Matrix multiplication of size 1024 using 1024 threads
Matrix multiplication took 1.07 seconds

 Performance counter stats for './mm1.out 1024 1024':

          21375.86 msec task-clock                #   17.824 CPUs utilized
             10664      context-switches          #    0.499 K/sec
              2881      cpu-migrations            #    0.135 K/sec
              5380      page-faults               #    0.252 K/sec
       52419788161      cycles                    #    2.452 GHz
       63744138775      instructions              #    1.22  insn per cycle
        2225901485      branches                  #  104.132 M/sec
           1812376      branch-misses             #    0.08% of all branches

       1.199282973 seconds time elapsed

      21.232146000 seconds user
       0.192400000 seconds sys
```

Observation:

- As the number of threads increased:
  - Task-clock (CPU utilized) increased but eventually plateau.
  - Context-switches increased in general, but inconsistent, e.g. decreased from 32 to 64 threads but increased back from 128 threads.
  - CPU-migrations increased.
  - Page faults increased, and the difference was small initially but it was slowly getting bigger.
  - Cycles increased initially but plateau very quickly.
  - Instructions per cycle remained constant.
  - Branches remained constant.
  - Branch-misses remained constant.
  - Time elapsed decreased but eventually plateau.
  - User time increased but eventually plateau.
  - Sys time remained constant.

## Part 3: Running and profiling compute-heavy applications with Slurm

### Exercise 5

```shell
$ ./run_job.sh 1024 8
Matrix multiplication took 1.85 seconds
Matrix multiplication of size 1024 using 8 threads

 Performance counter stats for '/home/richwill/mm1_job//mm1.out 1024 8':

          14542.01 msec task-clock                #    7.662 CPUs utilized
              1120      context-switches          #    0.077 K/sec
                10      cpu-migrations            #    0.001 K/sec
              3192      page-faults               #    0.220 K/sec
       58038160751      cycles                    #    3.991 GHz
       63638886830      instructions              #    1.10  insn per cycle
        2206426933      branches                  #  151.728 M/sec
           1677215      branch-misses             #    0.08% of all branches

       1.897885041 seconds time elapsed

      14.536121000 seconds user
       0.008015000 seconds sys

$ ./run_job.sh 1024 16
Matrix multiplication took 1.76 seconds
Matrix multiplication of size 1024 using 16 threads

 Performance counter stats for '/home/richwill/mm1_job//mm1.out 1024 16':

          13557.11 msec task-clock                #    7.540 CPUs utilized
              2525      context-switches          #    0.186 K/sec
                85      cpu-migrations            #    0.006 K/sec
              3208      page-faults               #    0.237 K/sec
       54116380764      cycles                    #    3.992 GHz
       63612107926      instructions              #    1.18  insn per cycle
        2198366549      branches                  #  162.156 M/sec
           1360246      branch-misses             #    0.06% of all branches

       1.798015182 seconds time elapsed

      13.541706000 seconds user
       0.019996000 seconds sys

$ ./run_job.sh 1024 32
Matrix multiplication took 1.73 seconds
Matrix multiplication of size 1024 using 32 threads

 Performance counter stats for '/home/richwill/mm1_job//mm1.out 1024 32':

          13821.13 msec task-clock                #    7.816 CPUs utilized
              1747      context-switches          #    0.126 K/sec
               116      cpu-migrations            #    0.008 K/sec
              3242      page-faults               #    0.235 K/sec
       55156994465      cycles                    #    3.991 GHz
       63605807904      instructions              #    1.15  insn per cycle
        2197000442      branches                  #  158.960 M/sec
           1246671      branch-misses             #    0.06% of all branches

       1.768418576 seconds time elapsed

      13.818439000 seconds user
       0.004000000 seconds sys

$ ./run_job.sh 1024 64
Matrix multiplication of size 1024 using 64 threads
Matrix multiplication took 1.73 seconds

 Performance counter stats for '/home/richwill/mm1_job//mm1.out 1024 64':

          13806.53 msec task-clock                #    7.833 CPUs utilized
              3419      context-switches          #    0.248 K/sec
               215      cpu-migrations            #    0.016 K/sec
              3311      page-faults               #    0.240 K/sec
       55100165935      cycles                    #    3.991 GHz
       63614098115      instructions              #    1.15  insn per cycle
        2198739370      branches                  #  159.254 M/sec
           1281456      branch-misses             #    0.06% of all branches

       1.762549060 seconds time elapsed

      13.804289000 seconds user
       0.004003000 seconds sys

$ ./run_job.sh 1024 128
Matrix multiplication took 1.72 seconds
Matrix multiplication of size 1024 using 128 threads

 Performance counter stats for '/home/richwill/mm1_job//mm1.out 1024 128':

          13768.61 msec task-clock                #    7.837 CPUs utilized
              4008      context-switches          #    0.291 K/sec
               341      cpu-migrations            #    0.025 K/sec
              3449      page-faults               #    0.250 K/sec
       54948492627      cycles                    #    3.991 GHz
       63622417236      instructions              #    1.16  insn per cycle
        2200450726      branches                  #  159.817 M/sec
           1306899      branch-misses             #    0.06% of all branches

       1.756843930 seconds time elapsed

      13.747000000 seconds user
       0.023963000 seconds sys

$ ./run_job.sh 1024 256
Matrix multiplication took 1.73 seconds
Matrix multiplication of size 1024 using 256 threads

 Performance counter stats for '/home/richwill/mm1_job//mm1.out 1024 256':

          13714.86 msec task-clock                #    7.729 CPUs utilized
              7298      context-switches          #    0.532 K/sec
               870      cpu-migrations            #    0.063 K/sec
              3724      page-faults               #    0.272 K/sec
       54713038317      cycles                    #    3.989 GHz
       63657689192      instructions              #    1.16  insn per cycle
        2207757509      branches                  #  160.976 M/sec
           1593303      branch-misses             #    0.07% of all branches

       1.774458394 seconds time elapsed

      13.661361000 seconds user
       0.064137000 seconds sys

$ ./run_job.sh 1024 512
Matrix multiplication took 1.72 seconds
Matrix multiplication of size 1024 using 512 threads

 Performance counter stats for '/home/richwill/mm1_job//mm1.out 1024 512':

          13779.32 msec task-clock                #    7.792 CPUs utilized
              5919      context-switches          #    0.430 K/sec
              1162      cpu-migrations            #    0.084 K/sec
              4275      page-faults               #    0.310 K/sec
       54981626053      cycles                    #    3.990 GHz
       63669204346      instructions              #    1.16  insn per cycle
        2209973460      branches                  #  160.383 M/sec
           1463981      branch-misses             #    0.07% of all branches

       1.768390537 seconds time elapsed

      13.747231000 seconds user
       0.039939000 seconds sys

$ ./run_job.sh 1024 1024
Matrix multiplication took 1.70 seconds
Usage: /home/richwill/mm1_job//mm1.out <size> <threads>
Matrix multiplication of size 1024 using 1024 threads

 Performance counter stats for '/home/richwill/mm1_job//mm1.out 1024 1024':

          13511.48 msec task-clock                #    7.682 CPUs utilized
             11466      context-switches          #    0.849 K/sec
              2947      cpu-migrations            #    0.218 K/sec
              5381      page-faults               #    0.398 K/sec
       53889080902      cycles                    #    3.988 GHz
       63753609460      instructions              #    1.18  insn per cycle
        2227408941      branches                  #  164.853 M/sec
           1874262      branch-misses             #    0.08% of all branches

       1.758788018 seconds time elapsed

      13.433773000 seconds user
       0.099775000 seconds sys
```

Note: The jobs are executed on i7-7700.

Observation:

- As the number of threads increased:
  - Task-clock (CPU utilized) remained constant at around 7.x, probably because it is bounded by the number of cores (e.g., i7-7700 has only 8 cores).
  - Context-switches increased in general, but inconsistent.
  - CPU-migrations increased.
  - Page faults increased, and the difference was small initially but it was slowly getting bigger.
  - Cycles remained constant.
  - Instructions per cycle remained constant.
  - Branches remained constant.
  - Branch-misses remained constant.
  - Time elapsed decreased but eventually plateau.
  - User time remained constant.
  - Sys time remained constant.

Comparision with Exercise 3:

- Mostly similar results. The most noticable difference is the user time which remained constant throughout with Slurm.
