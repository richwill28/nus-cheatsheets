/**
 * CS3210 - Non-blocking communication in MPI.
 */

#include "mpi.h"
#include <stdio.h>
#include <unistd.h>
#include <string.h>

int main(int argc, char *argv[])
{
  int numtasks, rank, next, prev, tag1=1, tag2=2;
  int buf[2];
  char hostname[256];
  
  MPI_Request reqs[4];
  MPI_Status stats[4];

  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  memset(hostname, 0, sizeof(hostname));
  int sc_status = gethostname(hostname, sizeof(hostname)-1);
  if (sc_status)
  {
    perror("gethostname");
    return sc_status;
  }

  prev = rank-1;
  next = rank+1;
  if (rank == 0)
    prev = numtasks - 1;
  
  if (rank == (numtasks - 1))
    next = 0;

  MPI_Isend(&rank, 1, MPI_INT, prev, tag2, MPI_COMM_WORLD, &reqs[2]);
  MPI_Isend(&rank, 1, MPI_INT, next, tag1, MPI_COMM_WORLD, &reqs[3]);

  MPI_Irecv(&buf[0], 1, MPI_INT, prev, tag1, MPI_COMM_WORLD, &reqs[0]);
  MPI_Irecv(&buf[1], 1, MPI_INT, next, tag2, MPI_COMM_WORLD, &reqs[1]);
  
  MPI_Waitall(4, reqs, stats);

  printf("rank=%d (on %s) received value=%d from rank=%d\n", rank, hostname, buf[0], prev);
  printf("rank=%d (on %s) received value=%d from rank=%d\n", rank, hostname, buf[1], next);

  MPI_Finalize();
  
  return 0;
}

