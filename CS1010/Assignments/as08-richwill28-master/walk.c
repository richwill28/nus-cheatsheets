/**
 * CS1010 Semester 1 AY20/21
 * Assignment 8: Walk
 *
 * This program prints the number of possible paths we can walk
 * to the destination that is x block to the east and y block to
 * the north.
 *
 * @file: walk.c
 * @author: Richard Willie (Group C08)
 */
#include "cs1010.h"

/**
 * Print the number of possible paths in an x * y grid.
 *
 * @param[in,out] grid An x * y grid.
 * @param[in] x The number of rows in the grid.
 * @param[in] y The number of columns in the grid.
 */
void path(long **grid, long x, long y) {
  /*
   Symmetry: traversing east-north from bottom-left of the grid is
   the same as traversing east-south from upper-left of the grid.
   */
  for (long i = 0; i <= x; i += 1) {
    for (long j = 0; j <= y; j += 1) {
      if (i == 0 || j == 0) {
        grid[i][j] = 1;
      } else {
        grid[i][j] = grid[i][j - 1] + grid[i - 1][j];
      }
    }
  }
  cs1010_println_long(grid[x][y]);
}

int main() {
  // Read the number of rows and columns from the standard inputs. 
  long x = cs1010_read_long();
  long y = cs1010_read_long();

  // Initialize an x * y grid and check for memory allocation error.
  long **grid = malloc((size_t)(x + 1) * sizeof(long *));
  if (grid == NULL) {
    return -1;
  }
  
  for (long i = 0; i <= x; i += 1) {
    grid[i] = malloc((size_t)(y + 1) * sizeof(long));
    if (grid[i] == NULL) {
      return -1;
    }
  }
  
  // Print the number of possible paths in an x * y grid.
  path(grid, x, y);

  // Release the memory allocated for the grid.
  for (long i = 0; i <= x; i += 1) {
    free(grid[i]);
  }
  free(grid);
  return 0;
}
