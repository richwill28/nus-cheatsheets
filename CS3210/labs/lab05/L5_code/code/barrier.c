/**
 * CS3210 - Collective communication in MPI.
 */

#include <mpi.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#define ITERATIONS 100

int main(int argc, char *argv[]) {
  MPI_Init(&argc,&argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int master_node_rank = size - 1;
  int num_workers = size - 1;

  // Timing info
  double loop_start_time_s = MPI_Wtime();

  int number = rank;
  if (rank == master_node_rank) {
    // Master node
    for (int i = 0; i < num_workers * ITERATIONS; i++) {
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Recv(&number, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      double print_delay = MPI_Wtime() - loop_start_time_s;
      printf("Master node received number %d at time %.5f sec\n", number, print_delay);
    }
  } else {
    // Workers
    for (int i = 0; i < num_workers  * ITERATIONS; i++) {
      if (rank == (i % num_workers)) {
        // Only this particular worker should send in this iteration
        MPI_Send(&rank, 1, MPI_INT, master_node_rank, 0, MPI_COMM_WORLD);
      }

      MPI_Barrier(MPI_COMM_WORLD);

      // Random sleep to vary the workers
      useconds_t sleepTime = (useconds_t)(((rand() % 5) + 1) * 100);
      usleep(sleepTime);
    }
  }

  MPI_Finalize();
  return 0;
}
