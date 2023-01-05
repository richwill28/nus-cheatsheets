## Part 1: Hello World in CUDA

### Exercise 1

How many threads are there per block? How many blocks are in the grid?

- N threads per block.
- 1 block is in the grid.

How many threads are there in total?

- N threads.

### Exercise 2

Modify hello.cu such that it runs on a 3D grid of dimension 2 × 2 × 2, where each block has a dimension of 2 × 4.

Answer:

- 8 block is in the grid.
- 8 threads per block.
- Block 0 handles idx 0 to 7
- Block 1 handles idx 8 to 15
- ...
- Block 7 handles idx 56 to 63
- [Flatten the 3D grid and 2D block to map to a single index.](https://stackoverflow.com/questions/7367770/how-to-flatten-or-index-3d-array-in-1d-array)

### Exercise 3

Compile and execute `slow.cu`. Notice the extremely slow runtime when each block contains 1 thread. Calculate the number of warps launched when running 1024 blocks of 1 thread each, versus 1 block of 1024 threads.

Answer:

- 1024 blocks of 1 thread: 1024 warps
- 1 block of 1024 threads: 32 warps

### Exercise 4

Compile and execute `printing.cu`. Notice that there is a trend in how the thread IDs are printed out. Notice any correlation with our knowledge on how the GPU executes a CUDA program?

Answer:

- The thread IDs are printed out in group of 32.

## Part 2: CUDA Memory Model

### Exercise 5

Compile and run `global_comm.cu`. Observe how the variable result can be accessed by both the device and host (albeit indirectly). What's the reason behind this indirect access?

### Exercise 7

Compile and run both `global_comm_unified.cu` and `cudaMalloc_comm_unified.cu` (Note that you may need to tell nvcc to target architecture sm 70 or below). Compare the code with their non-unified counterparts. What difference(s) do you observe?

Answer:

- Host can directly access the variable in unified memory.

### Exercise 8

Compile and run `shared_mem.cu`. Observe/Ponder on the following:

- Are there any differences between shared and global memory?
- Do the results printed out differ between runs?

Answer:

- For a given block, the results between shared and global memory are different.
- However, within the same block two threads have the same results for both shared and global memory.
- The results with global memory differ between runs.

## Part 3: Synchronization in CUDA

### Exercise 9

Compile and run atomic.cu. Observe on the following:

- What are the values that are printed out? Are they consistent across different runs?
- How does the code resolve global memory access race condition?

Answer:

- The result is consistent when `atomicAdd()` is used.

### Exercise 10

Compile and run `synchronise.cu`. Observe/Ponder on the following:

- What is the significance of counter values printed out with/without syncthreads
- Why does the values vary when `__syncthreads` is used in a kernel launch containing multiple blocks?
- Why do you think the `is_done` variable is marked as volatile?

Answer:

- `__syncthreads` only synchronize the threads that belong to the same block.
- The use of volatile is to prevent compiler optimization. Before the while loop, the compiler thought that `is_done` will not be mutated (as there is no write, only read). Due to compiler optimization, the value of `is_done` will be accessed directly from the register, not from the memory. Therefore, if we remove the volatile, the program will stuck in an infinite loop.
