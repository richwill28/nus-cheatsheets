/**
 * CS1010 Semester 1 AY20/21
 * Assignment 5: Contact
 *
 * A program to determine whether two random people have
 * a direct contact, an indirect contact, or no contact.
 * In this program, contact is assumed to be bi-directional.
 * "Direct contact" and "no contact" are denoted by '1' and
 * '0' respectively.
 *
 * @file: contact.c
 * @author: Richard Willie (Group C08)
 */
#include "cs1010.h"
#include <stdbool.h>

/**
 * A function to check whether two people have a direct
 * contact.
 *
 * @param[in] jagged A jagged array containing information
 *                   of direct contacts between people.
 * @param[in] m The person 'm'.
 * @param[in] n The person 'n'.
 *
 * @return Returns true or false.
 */
bool is_direct(char **jagged, long m, long n) {
  bool value = false;

  if (m >= n && jagged[m][n] == '1') {
    value = true;
  } else if (m < n && jagged[n][m] == '1') {
    value = true;
  }

  return value;
}

/**
 * Check for indirect contact between two people. If there
 * exist an indirect contact through the person x, then
 * print 'contact through x', otherwise print 'no contact'.
 * Do note that if there exist multiple indirect contacts,
 * print the one with the smallest id.
 *
 * @param[in] jagged A jagged array contact containing
 *                   of direct contacts between people.
 * @param[in] num_of_rows The number of rows of the jagged
 *                        array.
 * @param[in] m The person 'm'.
 * @param[in] n The person 'n'.
 */
void indirect(char **jagged, long num_of_rows, long m, long n) {
  /*
   Flag variable to prevent the algorithm from checking an
   indirect contact between two people more than once as we
   only need to print the person with the smallest id.
   */
  long flag = 0;

  /*
   Iterate through each rows of the array to find an
   indirect contact.
   */
  for (long i = 0; i <= num_of_rows - 1; i += 1) {
    if (flag == 0) {
      if (m > i && jagged[m][i] == '1') {
        if (n > i && jagged[n][i] == '1') {
          cs1010_print_string("contact through ");
          cs1010_println_long(i);
          flag += 1;
        } else if (n < i && jagged[i][n] == '1') {
          cs1010_print_string("contact through ");
          cs1010_println_long(i);
          flag += 1;
        }
      } else if (m < i && jagged[i][m] == '1') {
        if (n > i && jagged[n][i] == '1') {
          cs1010_print_string("contact through ");
          cs1010_println_long(i);
          flag += 1;
        } else if (n < i && jagged[i][n] == '1') {
          cs1010_print_string("contact through ");
          cs1010_println_long(i);
          flag += 1;
        }
      }
    }
  }

  // If there is no indirect contact, print 'no contact'.
  if (flag == 0) {
    cs1010_println_string("no contact");
  }
}

int main() {
  // Read the number of rows of the jagged array.
  long num_of_rows = cs1010_read_long();

  // Initialize the jagged array.
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

  // Read the first person's id.
  long m = cs1010_read_long();

  // Read the second person's id.
  long n = cs1010_read_long();

  /*
   If m and n have a direct contact, print 'direct contact',
   otherwise run the 'indirect' function.
   */
  if (is_direct(jagged, m, n)) {
    cs1010_println_string("direct contact");
  } else {
    indirect(jagged, num_of_rows, m, n);
  }

  // Release the memory allocated for the jagged array.
  for (long i = 0; i <= num_of_rows - 1; i += 1) {
    free(jagged[i]);
  }
  free(jagged);

  return 0;
}
