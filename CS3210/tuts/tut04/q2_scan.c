#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <unistd.h>
#define TAG_LEFT 0
#define TAG_RIGHT  1
#define SIZE 200
int main (int argc, char ** argv)  {
	int rank, p, size=SIZE;
	int left, right;

	int input[SIZE];
	int prefix[SIZE] = {0};
	int my_num = 0;

	MPI_Init (&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	left = rank - 1;
	right = rank + 1;

	if (p != SIZE) {
		printf("Exiting: need number of ranks (%d) = number of input ints (%d)\n", p, SIZE);
		MPI_Abort(MPI_COMM_WORLD, 1);
		exit(1);
	}


	// Only rank 0 initializes the array
	if (rank == 0) 
		for (int i = 0; i < SIZE; i++) 
			input[i] = i;

	// Scatter array to everyone as 1 piece each
	MPI_Scatter(input, 1, MPI_INT, &my_num, 1, MPI_INT, 0, MPI_COMM_WORLD);
	printf("Rank %d: my_num is %d\n", rank, my_num);


	// Receive prefix sums from left if we are not the root process
	if (rank != 0) {
		MPI_Recv(prefix, SIZE, MPI_INT, left, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	// Compute our prefix sum
	int prev_num = rank != 0 ? prefix[rank - 1] : 0;
	int my_sum = prev_num + my_num;
	prefix[rank] = my_sum;

	// Send it to the right if we are not the last rank
	if (rank != (p - 1)) {
		MPI_Send(prefix, SIZE, MPI_INT, right, 0, MPI_COMM_WORLD);
	}

	// Final rank gets all the prefix sums and can print them out
	if (rank == (p - 1)) {
		printf("Prefix sums: ");
		for (int i = 0; i < SIZE; i++) {
			printf("%d, ", prefix[i]);
		}
		printf("\n");
	}

	MPI_Finalize();
	return EXIT_SUCCESS;
}

