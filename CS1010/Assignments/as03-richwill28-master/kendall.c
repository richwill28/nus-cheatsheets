/**
 * CS1010 Semester 1 AY20/21
 * Assignment 3: Kendall
 *
 * Read in a sequence of numbers and find its kendall tau 
 * distance.
 *
 * @file: kendall.c
 * @author: Richard Willie (Group C08)
 */
#include "cs1010.h"

double kendall(long list[], long n) {
  // Kendall tau distance
  double tau = 0;
  
  // number of comparison
  double comp = 0;

  // compare every possible pair of values
  for (long i = n - 2; i >= 0; i -= 1) {
    for (long j = i + 1; j <= n - 1; j += 1) {
      if (list[i] > list[j]) {
        tau += 1;
      }
      comp += 1;
    }
  }

  // normalized Kendall tau distance
  double norm = tau / comp;

  return norm;
}


int main() {
  long n = cs1010_read_long();

  long *values = cs1010_read_long_array(n);
  if (values == NULL) {
    return 1;
  }
  
  cs1010_println_double(kendall(values, n));
  free(values);
  return 0;
}
