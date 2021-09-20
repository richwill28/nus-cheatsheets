/**
 * CS1010 Semester 1 AY20/21
 * Assignment 5: Life
 *
 * This program simulates "Game of Life", a game devised
 * by a British mathematician John H. Conway. The program
 * reads from the input, three positve numbers n (n > 2),
 * m (m > 2), and k, where n and m denote the number of
 * rows and columns of a matrix, and k is the number of
 * iterations to simulate. Then it reads an n x m matrix of
 * characters. The characters are either '#' (alive) or '.'
 * (dead).
 *
 * @file: life.c
 * @author: Richard Willie (Group C08)
 */
#include "cs1010.h"
#include <unistd.h>

/**
 * Read the position of the character in the matrix and
 * output the number of alive neighbors.
 *
 * @param[in] matrix The n x m matrix.
 * @param[in] x The row location of the character.
 * @param[in] y The column location of the character.
 * 
 * @return Returns the number of alive neighbors.
 */
long live_neighbor(char **matrix, long x, long y) {
  long alive = 0;

  for (long i = x - 1; i <= x + 1; i += 1) {
    for (long j = y - 1; j <= y + 1; j += 1) {
      if (!(i == x && j == y) && matrix[i][j] == '#') {
        alive += 1;
      }
    }
  }

  return alive;
}

/**
 * A function to simulates "Game of Life" given an initial
 * matrix of characters.
 *
 * @param[in,out] matrix The n x m matrix.
 * @param[in,out] new_matrix The duplicate matrix.
 * @param[in] n The number of rows of the matrix.
 * @param[in] m The number of columns of the matrix.
 *
 * @pre All characters at the border of the matrix are dead.
 * @post All characters at the border of the matrix are dead.
 */
void life(char **matrix, char **new_matrix, long n, long m) {
  /*
   Iterate through the characters in the matrix. If the
   character is alive and has less than 2 or more than 3
   alive neighbors, then change said character to 'dead'.
   On the other hand, if the character is dead and has
   exactly 3 alive neighbors, then change said character
   to 'alive'.
   */
  for (long i = 1; i <= n - 2; i += 1) {
    for (long j = 1; j <= m - 2; j += 1) {
      long alive = live_neighbor(matrix, i, j);
      if (matrix[i][j] == '#' && (alive <= 1 || alive >= 4)) {
        new_matrix[i][j] = '.';
      } else if (matrix[i][j] == '.' && alive == 3) {
        new_matrix[i][j] = '#';
      }
    }
  }

  // Print the updated matrix.
  for (long i = 0; i <= n - 1; i += 1) {
    for (long j = 0; j <= m - 1; j += 1) {
      putchar(new_matrix[i][j]);
    }
    cs1010_println_string("");
  }

  // Set matrix to the updated matrix.
  for (long i = 0; i <= n - 1; i += 1) {
    for (long j = 0; j <= m - 1; j += 1) {
      matrix[i][j] = new_matrix[i][j];
    }
  }
}

int main() {
  /*
   Initialize n (number of rows), m (number of columns),
   and k (number of iterations).
   */
  long n = cs1010_read_long();
  long m = cs1010_read_long();
  long k = cs1010_read_long();

  // Initialize an n x m matrix.
  char **matrix;
  matrix = malloc((size_t)n * sizeof(char *));

  // Check for memory allocation error.
  if (matrix == NULL) {
    return -1;
  }

  // Read the matrix from the input.
  for (long i = 0; i <= n - 1; i += 1) {
    matrix[i] = cs1010_read_line();
  }

  // Simulates "Game of Life" for k iteration(s).
  for (long i = 1; i <= k; i += 1) {
    // Initialize a new matrix.
    char **new_matrix;
    new_matrix = malloc((size_t)n * sizeof(char *));

    // Check for memory allocation error.
    if (new_matrix == NULL) {
      return -1;
    }

    for (long j = 0; j <= n - 1; j += 1) {
      new_matrix[j] = malloc((size_t)m * sizeof(char));

      // Check for memory allocation error
      if (new_matrix[j] == NULL) {
        return -1;
      }
    }

    // Copy 'matrix' to 'new_matrix'
    for (long j = 0; j <= n - 1; j += 1) {
      for (long l = 0; l <= m - 1; l += 1 ) {
        new_matrix[j][l] = matrix[j][l];
      }
    }

    // Run "Game of Life".
    cs1010_clear_screen();
    life(matrix, new_matrix, n, m);
    usleep(250*1000);

    // Release the memory allocated for 'new_matrix'.
    for (long j = 0; j <= n - 1; j += 1) {
      free(new_matrix[j]);
    }
    free(new_matrix);
  }

  // Release the memory allocated for 'matrix'.
  for (long i = 0; i <= n - 1; i += 1) {
    free(matrix[i]);
  }
  free(matrix);

  return 0;
}
