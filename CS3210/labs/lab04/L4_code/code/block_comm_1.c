/**
 * CS3210 - Blocking communication in MPI.
 */

#include <mpi.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>

int main(int argc,char *argv[])
{
  int numtasks, rank, dest, source, rc, count, tag=1;  
  // char inmsg, outmsg='x';
  char hostname[256];
  MPI_Status Stat;

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

  if (rank == 0)	{
    dest = 1;
    source = 1;
    float inmsg[10];
    char outmsg = 'x';
    rc = MPI_Send(&outmsg, 1, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
    rc = MPI_Recv(&inmsg, 10, MPI_FLOAT, source, tag, MPI_COMM_WORLD, &Stat);
    rc = MPI_Get_count(&Stat, MPI_FLOAT, &count);
    printf("Task %d on %s: Received %d float(s) from task %d with tag %d \n",
            rank, hostname, count, Stat.MPI_SOURCE, Stat.MPI_TAG);
  } else if (rank == 1)	{
    dest = 0;
    source = 0;
    char inmsg;
    float outmsg[10] = {3.1, 3.14, 3.141, 3.1415, 3.14159, 2.7, 2.71, 2.718, 2.7182, 2.71828};
    rc = MPI_Recv(&inmsg, 1, MPI_CHAR, source, tag, MPI_COMM_WORLD, &Stat);
    rc = MPI_Send(&outmsg, 10, MPI_FLOAT, dest, tag, MPI_COMM_WORLD);
    rc = MPI_Get_count(&Stat, MPI_CHAR, &count);
    printf("Task %d on %s: Received %d char(s) from task %d with tag %d \n",
            rank, hostname, count, Stat.MPI_SOURCE, Stat.MPI_TAG);
  }

  MPI_Finalize();
  
  return 0;
}
