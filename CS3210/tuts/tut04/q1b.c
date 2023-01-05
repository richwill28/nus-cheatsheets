#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <unistd.h>
#define TAG_LEFT 0
#define TAG_RIGHT  1
#define SIZE 20000
int main (int argc, char ** argv)  {
	int rank, p, size=SIZE;
	int left, right;
	char send_buffer1[SIZE], recv_buffer1[SIZE];
	char send_buffer2[SIZE], recv_buffer2[SIZE];

	MPI_Status status;
	gethostname (send_buffer1, size);
	gethostname (send_buffer2, size);

	MPI_Init (&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	left = (rank-1+p) %p;
	right = (rank+1) %p;

	printf("Rank %d trying to send to %d\n", rank, left);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Send(send_buffer1, size, MPI_CHAR, left,
		TAG_LEFT, MPI_COMM_WORLD);
	sleep(1);
	MPI_Recv(recv_buffer1, size, MPI_CHAR, right,
		TAG_LEFT, MPI_COMM_WORLD, &status);

	MPI_Send(send_buffer2, size, MPI_CHAR, right,
		TAG_RIGHT, MPI_COMM_WORLD);
	MPI_Recv(recv_buffer2, size, MPI_CHAR, left,
		 TAG_RIGHT, MPI_COMM_WORLD, &status);

	MPI_Barrier(MPI_COMM_WORLD);
	printf ("my name (rank = %d): %s; left neighbors name: %s; right neighbors name: %s\n", rank, send_buffer1, recv_buffer1, recv_buffer2);
	MPI_Finalize();
	return EXIT_SUCCESS;
}

