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
	char inmsg, outmsg='x';
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
		rc = MPI_Send(&outmsg, 1, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
		rc = MPI_Recv(&inmsg, 1, MPI_CHAR, source, tag, MPI_COMM_WORLD, &Stat);
		
	} else if (rank == 1)	{
		dest = 0;
		source = 0;
		rc = MPI_Recv(&inmsg, 1, MPI_CHAR, source, tag, MPI_COMM_WORLD, &Stat);
		rc = MPI_Send(&outmsg, 1, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
	}

	rc = MPI_Get_count(&Stat, MPI_CHAR, &count);
	printf("Task %d on %s: Received %d char(s) from task %d with tag %d \n",
       		rank, hostname, count, Stat.MPI_SOURCE, Stat.MPI_TAG);

	MPI_Finalize();
	
	return 0;
}

