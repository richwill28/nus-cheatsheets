/**
 * CS3210 - Matrix Multiplication in MPI
 **/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <mpi.h>
#include <unistd.h>
#include <string.h>

int size;
int workers;
int myid;
char hostname[256];

long long comm_time = 0;
long long comp_time = 0;

#define MASTER_ID workers

typedef struct
{
	float ** element;
} matrix;


/** 
 * Determines the current time
 *
 **/
long long wall_clock_time()
{
#ifdef __linux__
	struct timespec tp;
	clock_gettime(CLOCK_REALTIME, &tp);
	return (long long)(tp.tv_nsec + (long long)tp.tv_sec * 1000000000ll);
#else
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (long long)(tv.tv_usec * 1000 + (long long)tv.tv_sec * 1000000000ll);
#endif
}

/**
 * Allocates memory for a matrix of size SIZE
 * The memory is allocated row-major order, i.e. 
 *  elements from the same row are allocated at contiguous 
 *  memory addresses.
 **/
void allocate_matrix(matrix* m, int rows, int columns)
{
	int i, j;

	// allocate array for all the rows
	m->element = (float**)malloc(sizeof(float*) * rows);
	if (m->element == NULL)
	{
		fprintf(stderr, "Out of memory\n");
		exit(1);
	}

	// allocate an array for each row of the matrix
	for	(i = 0; i < rows; i++)
	{
		m->element[i] = (float*)malloc(sizeof(float) * columns);
		if (m->element[i] == NULL)
		{
			fprintf(stderr, "Out of memory\n");
			exit(1);
		}
	}
}

/**
 * Initializes the elements of the matrix with
 * random values between 0 and 9
 **/
void init_matrix(matrix m)
{
	int i, j;

	for (i = 0; i < size; i++)
		for (j = 0; j < size; j++)
		{
			m.element[i][j] = rand() % 10;
		}
}

/**
 * Initializes the elements of the matrix with
 * element 0.
 **/
void init_matrix_zero(matrix m)
{
	int i, j;

	for (i = 0; i < size; i++)
		for (j = 0; j < size; j++)
		{
			m.element[i][j] = 0.0;
		}
}

/**
 * Prints the matrix
 * 
 **/
void print_matrix(matrix m)
{
	int i, j;

	for (i = 0; i < size; i++)
	{
		printf("row =%4d: ", i);
		for (j = 0; j < size; j++)
			printf("%6.2f  ", m.element[i][j]);
		printf("\n");
	}
}


/*************************************************************************************************************************************/

/**
 * Function used by the workers to receive data from the master
 * Each worker receives the entire B matrix
 * and a number of rows from the A matrix
 **/
void worker_receive_data(matrix* b, matrix *a)
{
	int i, row_id;
	int rows_per_worker = size / workers ;
	MPI_Status status;
	long long before, after;
   	
  before = wall_clock_time();
  allocate_matrix (a, rows_per_worker, size); 

  MPI_Request reqs[4096];
  MPI_Status stats[4096];
	int rid = 0;

	// Getting a few rows of matrix A from the master
	for (i = 0; i < rows_per_worker; i++)
	{
		row_id = myid * rows_per_worker + i;
		// MPI_Recv(a->element[i], size, MPI_FLOAT, MASTER_ID, row_id, MPI_COMM_WORLD, &status);
		MPI_Irecv(a->element[i], size, MPI_FLOAT, MASTER_ID, row_id, MPI_COMM_WORLD, &reqs[rid++]);
	}
	fprintf(stderr," --- WORKER %d: Received row [%d-%d] of matrix A\n", myid, myid*rows_per_worker, row_id);
	after = wall_clock_time();
	comm_time += after - before;


	// Getting the entire B matrix from the master
   
	before = wall_clock_time();
	allocate_matrix(b, size, size);
	after = wall_clock_time();
	comp_time += after - before;

	before = wall_clock_time();
	fprintf(stderr," --- WORKER %d: Receiving all %d rows for matrix B...\n", myid, size);
	for (i = 0; i < size; i++)
	{
		// fprintf(stderr," --- WORKER %d: Received row [%d] of matrix B\n", myid, i);
		// MPI_Recv(b->element[i], size, MPI_FLOAT, MASTER_ID, i, MPI_COMM_WORLD, &status);
		MPI_Irecv(b->element[i], size, MPI_FLOAT, MASTER_ID, i, MPI_COMM_WORLD, &reqs[rid++]);
	}

	MPI_Waitall(rid, reqs, stats);

	fprintf(stderr," --- WORKER %d: Received matrix B\n", myid);
	after = wall_clock_time();
	comm_time += after - before;

}


/** 
 * Function used by the workers to compute the product
 * result = A x B
 **/
void worker_compute(matrix b, matrix a, matrix *result)
{
	int i, j, k;
	int rows_per_worker = size / workers ;
	long long before, after;

	before = wall_clock_time();
    allocate_matrix (result, rows_per_worker, size); 

	for (i = 0; i < rows_per_worker; i++)
	{
		for ( j = 0; j < size; j++)
		{
			result->element[i][j] = 0;
			for (k = 0; k < size; k++)
			{
				result->element[i][j] += a.element[i][k] * b.element[k][j];
			}
		}
	}
	after = wall_clock_time();
	comp_time += after - before;

	fprintf(stderr," --- WORKER %d: Finished the computations\n", myid);
}

/**
 * Function used by the workers to send the product matrix
 * back to the master
 **/
void worker_send_result(matrix result)
{
	int i;
	int rows_per_worker = size / workers ;
	long long before, after;

	before = wall_clock_time();
	for (i = 0; i < rows_per_worker; i++)
	{
		int row_id = myid * rows_per_worker + i;
		MPI_Send(result.element[i], size, MPI_FLOAT, MASTER_ID, row_id, MPI_COMM_WORLD);
	}
	after = wall_clock_time();
	comm_time += after - before;
	fprintf(stderr," --- WORKER %d: Sent the results back\n", myid);
}


/**
 * Main function called by workers
 *
 **/
void worker()
{
    
	int rows_per_worker = size / workers ;

	matrix row_a_buffer;
	matrix b;
	matrix result;
 

	
	// Receive data
	worker_receive_data(&b, &row_a_buffer);

	// Doing the computations
	worker_compute(b, row_a_buffer, &result);

	// Sending the results back
	worker_send_result(result);

	fprintf(stderr, " --- WORKER %d (on %s): communication_time=%6.2f seconds; computation_time=%6.2f seconds\n", myid, hostname, comm_time / 1000000000.0, comp_time / 1000000000.0);
}


/*************************************************************************************************************************************/

/**
 * Function called by the master to distribute 
 * rows from matrix A among different workers
 * and the entire matrix B to each worker
 **/
void master_distribute(matrix a, matrix b)
{
	int i, j, k;
	int worker_id = 1;

  MPI_Request reqs[4096];
  MPI_Status stats[4096];
	int rid = 0;

	// Matrix A is split into each chunks;
	// Each chunck has rows_per_worker rows
	int rows_per_worker = size / workers ;
	int row_start, row_end, row_id;

	fprintf(stderr," +++ MASTER : Distributing matrix A to workers: workers %d, rows_per_worker %d \n", workers, rows_per_worker);
	// Send the rows to each process
	for (worker_id = 0; worker_id < workers; worker_id++)
	{	
		row_start = worker_id * rows_per_worker;
		row_end = row_start + rows_per_worker;

		for (row_id = row_start; row_id < row_end; row_id++)
		{
			//int row_id = worker_id * rows_per_worker + i;
			float row_a_buffer[size];

			for (k = 0; k < size; k++)
			{
				row_a_buffer[k] = a.element[row_id][k];
			}
			// MPI_Send(row_a_buffer, size, MPI_FLOAT, worker_id, row_id, MPI_COMM_WORLD);

			// buffer might be overriden so this is buggy?
			MPI_Isend(row_a_buffer, size, MPI_FLOAT, worker_id, row_id, MPI_COMM_WORLD, &reqs[rid++]);
		}
		fprintf(stderr," +++ MASTER : Finished sending row [%d-%d] of matrix A to process %d\n", 
			row_start, row_end-1, worker_id);
	}

	// Send the entire B matrix to all workers
	fprintf(stderr," +++ MASTER : Sending matrix B to all workers\n");
	for (i = 0; i < size; i++)
	{
		float buffer[size];
		for (j = 0; j < size; j++)
			buffer[j] = b.element[i][j];

		for (worker_id = 0; worker_id < workers; worker_id++)
		{	
			// MPI_Send(buffer, size, MPI_FLOAT, worker_id, i, MPI_COMM_WORLD);
			MPI_Isend(buffer, size, MPI_FLOAT, worker_id, i, MPI_COMM_WORLD, &reqs[rid++]);
		}
	}

	MPI_Waitall(rid, reqs, stats);

	fprintf(stderr," +++ MASTER : Finished sending matrix B to all workers\n");
}

/**
 * Receives the result matrix from the workers
 * row by row and assembles it into the 
 * object @result
 **/
void master_receive_result(matrix result)
{
	int i, j, k;
	int worker_id = 1;
	MPI_Status status;

	fprintf(stderr," +++ MASTER: Receiving the results from workers\n");

	// Matrix a is distributed part by part
	int rows_per_worker = size / workers ;	
	// Get the results
	for (worker_id = 0; worker_id < workers ; worker_id++)
	{	
		for (i = 0; i < rows_per_worker; i++)
		{
			int row_id = worker_id * rows_per_worker + i;
			float buffer[size];
			MPI_Recv(buffer, size, MPI_FLOAT, worker_id, row_id, MPI_COMM_WORLD, &status);
			for (j = 0; j < size; j++)
				result.element[row_id][j] = buffer[j];
		}
	}
}

/**
 * Main function called by the master process
 *
 **/
void master()
{
	matrix a, b, result;
	long long before, after;

	// Start timer
	before = wall_clock_time();

	// Allocate memory for matrices
	allocate_matrix(&a, size, size);
	allocate_matrix(&b, size, size);
	allocate_matrix(&result, size, size);

	// Initialize matrix elements
	init_matrix(a);
	init_matrix(b);


	// Distribute data to workers
	master_distribute(a, b);

	// Gather results from workers
	master_receive_result(result);

	// End timer
	after = wall_clock_time();
	comp_time += after - before;
	fprintf(stderr, " --- MASTER %d (on %s): total time=%6.2f seconds\n", myid, hostname, comp_time / 1000000000.0);

	// Print the result matrix
	print_matrix(result);
}



/*************************************************************************************************************************************/

/**
 * Matrix multiplication using master-worker paradigm
 * The master initializes and sends the data to the 
 * workers. The workers do all the computation and send 
 * the product back to the master
 *
 * Matrix size must by a power of 2
 * Number of workers must be a power of 2 and less than matrix size
 * 
 * Total number of processes is 1 + number of workers
 *
 **/
int main(int argc, char ** argv)
{
	int nprocs;
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);


	// Set size based on arguments
	if (argc >= 2) 
	{
		size = atoi(argv[1]);
	} 
	else 
	{
		size = 2048;
	}

	// Get hostname for all prints
	memset(hostname, 0, sizeof(hostname));
	int sc_status = gethostname(hostname, sizeof(hostname)-1);
	if (sc_status)
	{
		perror("gethostname");
		return sc_status;
	}

	// One master and nprocs-1 workers
	workers = nprocs - 1;

	if (myid == MASTER_ID)
	{
		fprintf(stderr, "Usage: %s <size>\n", argv[0]);
		fprintf(stderr, " +++ Process %d (on %s) is master\n", myid, hostname);
		fprintf(stderr, "Multiplying matrix size %d x %d\n", size, size);
		master();
	}
	else
	{
		fprintf(stderr, " --- Process %d (on %s) is worker\n", myid, hostname);
		worker();
	}	
	MPI_Finalize();
	return 0;
}
