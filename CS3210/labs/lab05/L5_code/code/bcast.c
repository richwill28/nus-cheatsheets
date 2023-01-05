#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>

void my_bad_bcast(void* data, int count, int my_rank, int world_size) {
  if (my_rank == 0) {
    // If we are the root process, send our data to everyone
    int i;
    for (i = 0; i < world_size; i++) {
      if (i != my_rank) {
        MPI_Send(data, count, MPI_INT, i, 0, MPI_COMM_WORLD);
      }
    }
  } else {
    // If we are a receiver process, receive the data from the root
    MPI_Recv(data, count, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "Usage: %s num_elements num_trials\n", argv[0]);
    exit(1);
  }

  int num_elements = atoi(argv[1]);
  int num_trials = atoi(argv[2]);

  MPI_Init(NULL, NULL);

  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  double total_my_bad_bcast_time = 0.0;
  double total_mpi_bcast_time = 0.0;
  int i;
  int* data = (int*)malloc(sizeof(int) * num_elements);
  assert(data != NULL);

  for (i = 0; i < num_trials; i++) {
    // Time my_bad_bcast
    // Synchronize before starting timing
    MPI_Barrier(MPI_COMM_WORLD);
    total_my_bad_bcast_time -= MPI_Wtime();
    my_bad_bcast(data, num_elements, rank, world_size);
    // Synchronize again before obtaining final time
    MPI_Barrier(MPI_COMM_WORLD);
    total_my_bad_bcast_time += MPI_Wtime();

    // Time MPI_Bcast
    MPI_Barrier(MPI_COMM_WORLD);
    total_mpi_bcast_time -= MPI_Wtime();
  
    // TODO: Insert your MPI_Bcast call here!
    MPI_Bcast(data, num_elements, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    total_mpi_bcast_time += MPI_Wtime();
  }

  // Print timing information
  if (rank == 0) {
    printf("Data size = %d, Trials = %d\n", num_elements * (int)sizeof(int), num_trials);
    double my_bad_bcast_time = total_my_bad_bcast_time / num_trials;
    double mpi_bcast_time = total_mpi_bcast_time / num_trials;
    printf("Avg my_bad_bcast time =\t%lf sec\n", my_bad_bcast_time);
    printf("Avg MPI_Bcast time =\t%lf sec\n", mpi_bcast_time);
    printf("MPI_Bcast Speedup =\t%lfx\n", (my_bad_bcast_time / mpi_bcast_time));
  }

  free(data);
  MPI_Finalize();
}
