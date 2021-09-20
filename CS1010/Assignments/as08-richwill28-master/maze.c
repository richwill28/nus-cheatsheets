/**
 * CS1010 Semester 1 AY20/21
 * Assignment 8: Maze
 *
 * This program explores a maze and look for a way out. In addition,
 * it prints each sequence of steps to create an animation.
 *
 * @file: maze.c
 * @author: Richard Willie (Group C08)
 */

#include "cs1010.h"
#include <stdbool.h>
#include <string.h>
#include <unistd.h>

#define EMPTY     '.'
#define WALL      '#'
#define USER      '@'
#define VISITED   1
#define UNVISITED 0

/**
 * Print the maze and the number of steps.
 *
 * @param[in] maze The maze.
 * @param[in] rows The number of rows in the maze.
 * @param[in] steps The number of steps.
 */
void print_maze(char **maze, long rows, long *steps) {
  cs1010_clear_screen();
  for (long i = 0; i <= rows - 1; i += 1) {
    cs1010_println_string(maze[i]);
  }
  cs1010_println_long(*steps);

  // Sleep only if we are displaying in the terminal.
  if (isatty(fileno(stdout))) {
    usleep(100*1000);
  }
}

/**
 * Check whether a particular point is traversable or not.
 *
 * @param[in] maze The maze.
 * @param[in] status The status array.
 * @param[in] rows The number of rows in the maze.
 * @param[in] cols The number of cols in the maze.
 * @param[in] row The row position of the point.
 * @param[in] col The column position of the point.
 *
 * @return Return true if the point is traversable, otherwise false.
 */
bool is_traversable(char **maze, long **status, long rows, long cols, long row, long col) {
  return ((row >= 0 && row < rows) && (col >= 0 && col < cols) && maze[row][col] == EMPTY && status[row][col] == UNVISITED);
}

/**
 * Update the maze and print it.
 *
 * @param[in,out] maze The maze.
 * @param[in] rows The number of rows in the maze.
 * @param[in] row The row position of Scully.
 * @param[in] col The column position of Scully.
 * @param[in] new_row The new row position of Scully.
 * @param[in] new_col The new column position of Scully.
 * @param[in,out] steps The number of steps.
 */
void update_maze(char **maze, long rows, long row, long col, long new_row, long new_col, long *steps) {
  char temp = maze[row][col];
  maze[row][col] = maze[new_row][new_col];
  maze[new_row][new_col] = temp;
  *steps += 1;
  print_maze(maze, rows, steps);
}

/**
 * A strategy to explore the maze and find way out. The strategy is similar to a Depth-first search algorithm,
 * in which we traverse each point in the maze in the sequence up/right/down/left and explore as far as possible
 * along each branch before backtracking if no way out is to be found.
 *
 * @param[out] maze The maze.
 * @param[in,out] status The status array.
 * @param[in] rows The nunmber of rows in the maze.
 * @param[in] cols The number of columns in the maze.
 * @param[in] row The current row position of Scully.
 * @param[in] col The current column position of Scully.
 * @param[in] old_row The previous row position of Scully.
 * @param[in] old_col The previous column position of Scully.
 * @param[out] steps The number of steps.
 *
 * @return Return true if an exit is found, otherwise false.
 */
bool strategy(char **maze, long **status, long rows, long cols, long row, long col, long old_row, long old_col, long *steps) {
  // Update Scully's position.
  status[row][col] = VISITED;
  update_maze(maze, rows, old_row, old_col, row, col, steps);

  /* Base Case */
  if (row == 0 || row == rows - 1 || col == 0 || col == cols - 1) {
    return true;
  }
  
  /* Recursive Cases */

  // Move up and also update Scully's position when backtracking.
  if (is_traversable(maze, status, rows, cols, row - 1, col)) {
    if (strategy(maze, status, rows, cols, row - 1, col, row, col, steps)) {
      return true;
    }
    update_maze(maze, rows, row, col, row - 1, col, steps);
  }

  // Move right and also update Scully's position when backtracking.
  if (is_traversable(maze, status, rows, cols, row, col + 1)) {
    if (strategy(maze, status, rows, cols, row, col + 1, row, col, steps)) {
      return true;
    }
    update_maze(maze, rows, row, col, row, col + 1, steps);
  }

  // Move down and also update Scully's position when backtracking.
  if (is_traversable(maze, status, rows, cols, row + 1, col)) {
    if (strategy(maze, status, rows, cols, row + 1, col, row, col, steps)) {
      return true;
    }
    update_maze(maze, rows, row, col, row + 1, col, steps);
  }

  // Move left and also update Scully's position when backtracking.
  if (is_traversable(maze, status, rows, cols, row, col - 1)) {
    if (strategy(maze, status, rows, cols, row, col - 1, row, col, steps)) {
      return true;
    }
    update_maze(maze, rows, row, col, row, col - 1, steps);
  }

  return false;
}

/**
 * Find the initial position of Scully and explore the maze.
 *
 * @param[in,out] maze The maze.
 * @param[out] status The status array.
 * @param[in] rows The number of rows in the maze.
 * @param[in] cols The number of columns in the maze.
 * @param[out] steps The number of steps.
 */
void explore_maze(char **maze, long **status, long rows, long cols, long *steps) {
  // Find initial position.
  long row = 0;
  long col = 0;
  for (long i = 0; i < rows; i += 1) {
    for (long j = 0; j < cols; j += 1) {
      if (maze[i][j] == USER) {
        row = i;
        col = j;
      }
    }
  }

  // Execute a strategy to explore the maze and find a way out.
  strategy(maze, status, rows, cols, row, col, row, col, steps);
}

int main() {
  // Read the number of rows and columns from the standard inputs.
  long rows = cs1010_read_long();
  long cols = cs1010_read_long();
  
  // Read the maze from the standard input.
  char **maze = cs1010_read_word_array(rows);
  if (maze == NULL) {
    return -1;
  }

  // Initialize a 2D array to store the status of each traversable point in the maze.
  long **status = malloc((size_t)rows * sizeof(long *));
  if (status == NULL) {
    return -1;
  }

  for (long i = 0; i <= rows - 1; i += 1) {
    status[i] = calloc((size_t)cols, sizeof(long));
    if (status[i] == NULL) {
      return -1;
    }
  }
  
  // Initialize the number of steps.
  long steps = -1;

  // Explore the maze.
  explore_maze(maze, status, rows, cols, &steps);

  // Release the memory allocated for the maze.
  for (long i = 0; i <= rows - 1; i += 1) {
    free(maze[i]);
  }
  free(maze);

  // Release the memory allocated for the status array.
  for (long i = 0; i <= rows - 1; i += 1) {
    free(status[i]);
  }
  free(status);
  return 0;
}
