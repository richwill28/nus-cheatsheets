/**
 * CS1010 Semester 1 AY20/21
 * Assignment 3: CountingSort
 *
 * Read in a series of numbers between 0 and 10000 
 * and sort them using counting sort.
 *
 * @file: countingsort.c
 * @author: Richard Willie (Group C08)
 */
#include "cs1010.h"

long maxarr(long list[], long n) {
  // a function to calculate the maximum number in a given list
  long maxn = list[0];
  
  for (long i = 1; i <= n - 1; i += 1) {
    if (list[i] > maxn) {
      maxn = list[i];
    }
  }
  return maxn;
}

void countsort(long list[], long n) {
  long maxn = maxarr(list, n);

  // initialize an array to count the number of appearance of each element from a given list
  long count[10001] = {0};

  // iterate through the input's list and store the count value of each element
  for (long i = 0; i <= n - 1; i += 1) {
    count[list[i]] += 1;
  }

  // print the element and its number of appearance
  for (long i = 0; i <= maxn; i += 1) {
    if (count[i] != 0) {
      cs1010_print_long(i);
      cs1010_print_string(" ");
      cs1010_println_long(count[i]);
    }
  }

  // print the sorted array
  for (long i = 0; i <= maxn; i += 1) {
    if (count[i] != 0) {
      for (long j = 1; j <= count[i]; j += 1) {
        cs1010_println_long(i);
      }
    }
  }
}

int main() {
  long n = cs1010_read_long();

  long *values = cs1010_read_long_array(n);
  if (values == NULL) {
    return 1;
  }

  countsort(values, n);
  free(values);
  return 0;
}
