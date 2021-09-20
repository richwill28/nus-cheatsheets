/**
 * CS1010 Semester 1 AY20/21
 * Assignment 5: Social
 *
 * Reads from the standard input two positive integers n (number
 * of rows/people) and k (degree), followed by n lines of strings 
 * consisting of '1' or '0' representing the social network of
 * these n people. Then, the program prints to the standard output
 * the social network formed by the friendship chain of degree k.
 * Do note that friendship is assumed to be bi-directional.
 *
 * @file: social.c
 * @author: Richard Willie (Group C08)
 */
#include "cs1010.h"
#include <stdbool.h>

/**
 * Check if two people have a direct relationship.
 *
 * @param[in] jagged The jagged array consisting the social network.
 * @param[in] m The person 'm'.
 * @param[in] n The person 'n'.
 *
 * @return Return true or false.
 */
bool is_direct(char **jagged, long m, long n) {
  if (jagged[m][n] == '1') {
    return true;
  }
  return false;
}

/**
 * Find indirect relationships between two people. If they
 * have an indirect relationship for some degree k, then
 * we change their relationship to '1'.
 *
 * @param[in,out] jagged The jagged array consisting the social
 *                network.
 * @param[in] num_of_rows Number of rows.
 * @param[in] m The person 'm'.
 * @param[in] n The person 'n'.
 */
void indirect(char **jagged, long num_of_rows, long m, long n) {
  /*
   Iterate through each rows of the array to find an
   indirect relationship.
   */
  for (long i = 0; i <= num_of_rows - 1; i += 1) {
    if (m > i && jagged[m][i] == '1') {
      if (n > i && jagged[n][i] == '1') {
        jagged[m][n] = '1';
      } else if (n < i && jagged[i][n] == '1') {
        jagged[m][n] = '1';
      }
    } else if (m < i && jagged[i][m] == '1') {
      if (n > i && jagged[n][i] == '1') {
        jagged[m][n] = '1';
      } else if (n < i && jagged[i][n] == '1') {
        jagged[m][n] = '1';
      }
    }
  }
}

int main() {
  // Read the number of rows and degree from input.
  long num_of_rows = cs1010_read_long();
  long degree = cs1010_read_long();

  // Initialize a jagged array.
  char **jagged;
  jagged = malloc((size_t)num_of_rows * sizeof(char *));
  
  // Check for memory allocation error.
  if (jagged == NULL) {
    return -1;
  }

  // Read the jagged array from the input.
  for (long i = 0; i <= num_of_rows - 1; i += 1) {
    jagged[i] = cs1010_read_line();
  }

  // Repeat the 'indirect' function until the desired degree.
  for (long i = 2; i <= degree; i += 1) {
    for (long m = 0; m <= num_of_rows - 1; m += 1) {
      for (long n = 0; n <= m; n += 1) {
        if (!is_direct(jagged, m, n)) {
          indirect(jagged, num_of_rows, m, n);
        }
      }
    }
  }

  // Print the new array.
  for (long m = 0; m <= num_of_rows - 1; m += 1) {
    for (long n = 0; n <= m; n += 1) {
      putchar(jagged[m][n]);
    }
    cs1010_println_string("");
  }

  // Release the memory allocated for the jagged array.
  for (long i = 0; i <= num_of_rows - 1; i += 1) {
    free(jagged[i]);
  }
  free(jagged);

  return 0;
}
