/**
 * CS1010 Semester 1 AY20/21
 * Assignment 3: Max
 *
 * Read in a sequence of numbers and recursively find
 * the maximum.
 *
 * @file: max.c
 * @author: Richard Willie (Group C08)
 */
#include "cs1010.h"

long maxarr(const long list[], long start, long end) {
  /* Base Case */

  // if length of array is 1, return the element
  if (end == 0) {
    return list[0];
  }

  /* Recursive Case */

  // maximum value of the right portion of the list
  long maxr = list[end];

  // maximum value of the left portion of the list
  long maxl = maxarr(list, start, end - 1);

  if (maxr >= maxl) {
    return maxr;
  }
  return maxl;
}

int main() {
  long n = cs1010_read_long();
  long start = 0;
  long end = n - 1;
  
  long *values = cs1010_read_long_array(n);
  if (values == NULL) {
    return 1;
  }
  
  cs1010_println_long(maxarr(values, start, end));
  free(values);
  return 0;
}
