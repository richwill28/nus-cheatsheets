## Part 1: Collective Communication

### Exercise 2

Observations:

```shell
$ srun --ntasks=4 --nodes=2 /nfs/home/$USER/bcast 5000000 10

Data size = 20000000, Trials = 10
Avg my_bad_bcast time = 0.348455 sec
Avg MPI_Bcast time =    0.204662 sec
MPI_Bcast Speedup =     1.702589x

$ srun --ntasks=4 --nodes=3 /nfs/home/$USER/bcast 5000000 10

Data size = 20000000, Trials = 10
Avg my_bad_bcast time = 0.348989 sec
Avg MPI_Bcast time =    0.222900 sec
MPI_Bcast Speedup =     1.565676x

$ srun --ntasks=4 --nodes=4 /nfs/home/$USER/bcast 5000000 10

Data size = 20000000, Trials = 10
Avg my_bad_bcast time = 0.514840 sec
Avg MPI_Bcast time =    0.247935 sec
MPI_Bcast Speedup =     2.076510x
```

## Part 2: Managing Communicators

Output:

```shell
$ srun --ntasks=8 /nfs/home/$USER/new_comm

rank= 0 newrank= 0 recvbuf= 6
rank= 1 newrank= 1 recvbuf= 6
rank= 2 newrank= 2 recvbuf= 6
rank= 3 newrank= 3 recvbuf= 6
rank= 4 newrank= 0 recvbuf= 22
rank= 5 newrank= 1 recvbuf= 22
rank= 6 newrank= 2 recvbuf= 22
rank= 7 newrank= 3 recvbuf= 22
```

Rank 0 to 3 belong to a new group and communicator, so do rank 4 to 7.

## Part 3: Cartesian Virtual Topologies

### Exercise 6

Output:

```shell
$ srun --ntasks=16 /nfs/home/$USER/cart

rank= 0 coords= 0 0  neighbors(u,d,l,r)= -2 4 -2 1
rank= 0                 inbuf(u,d,l,r)= -2 4 -2 1

rank= 1 coords= 0 1  neighbors(u,d,l,r)= -2 5 0 2
rank= 1                 inbuf(u,d,l,r)= -2 5 0 2

rank= 2 coords= 0 2  neighbors(u,d,l,r)= -2 6 1 3
rank= 2                 inbuf(u,d,l,r)= -2 6 1 3

rank= 3 coords= 0 3  neighbors(u,d,l,r)= -2 7 2 -2
rank= 3                 inbuf(u,d,l,r)= -2 7 2 -2

rank= 4 coords= 1 0  neighbors(u,d,l,r)= 0 8 -2 5
rank= 4                 inbuf(u,d,l,r)= 0 8 -2 5

rank= 5 coords= 1 1  neighbors(u,d,l,r)= 1 9 4 6
rank= 5                 inbuf(u,d,l,r)= 1 9 4 6

rank= 6 coords= 1 2  neighbors(u,d,l,r)= 2 10 5 7
rank= 6                 inbuf(u,d,l,r)= 2 10 5 7

rank= 7 coords= 1 3  neighbors(u,d,l,r)= 3 11 6 -2
rank= 7                 inbuf(u,d,l,r)= 3 11 6 -2

rank= 8 coords= 2 0  neighbors(u,d,l,r)= 4 12 -2 9
rank= 8                 inbuf(u,d,l,r)= 4 12 -2 9

rank= 9 coords= 2 1  neighbors(u,d,l,r)= 5 13 8 10
rank= 9                 inbuf(u,d,l,r)= 5 13 8 10

rank= 10 coords= 2 2  neighbors(u,d,l,r)= 6 14 9 11
rank= 10                 inbuf(u,d,l,r)= 6 14 9 11

rank= 11 coords= 2 3  neighbors(u,d,l,r)= 7 15 10 -2
rank= 11                 inbuf(u,d,l,r)= 7 15 10 -2

rank= 12 coords= 3 0  neighbors(u,d,l,r)= 8 -2 -2 13
rank= 12                 inbuf(u,d,l,r)= 8 -2 -2 13

rank= 13 coords= 3 1  neighbors(u,d,l,r)= 9 -2 12 14
rank= 13                 inbuf(u,d,l,r)= 9 -2 12 14

rank= 14 coords= 3 2  neighbors(u,d,l,r)= 10 -2 13 15
rank= 14                 inbuf(u,d,l,r)= 10 -2 13 15

rank= 15 coords= 3 3  neighbors(u,d,l,r)= 11 -2 14 -2
rank= 15                 inbuf(u,d,l,r)= 11 -2 14 -2
```
