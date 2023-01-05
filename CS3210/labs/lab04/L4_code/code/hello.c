/**
 * CS3210 - Hello World in MPI
 **/

#include <stdio.h>
#include <mpi.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char **argv)
{
	int rank, size;
	char hostname[256];
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	memset(hostname, 0, sizeof(hostname));
	int sc_status = gethostname(hostname, sizeof(hostname)-1);
	if (sc_status)
	{
		perror("gethostname");
		return sc_status;
	}


	/* From here on, each process is free to execute its own code */

	printf("Hello world from process %d out of %d on host %s\n", rank, size, hostname);

	MPI_Finalize();

	return 0;
}
