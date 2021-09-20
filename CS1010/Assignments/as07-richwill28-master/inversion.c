/**
 * CS1010 Semester 1 AY20/21
 * Assignment 7: Inversion
 *
 * Count the amount of inverson of a given array. We assumed
 * the array to be first increasing and then decreasing. In
 * addition, all elements are distinct.
 *
 * @file: inversion.c
 * @author: Richard Willie (Group C08)
 */
#include "cs1010.h"

/**
 * Count the amount of inversion from a given array. The array
 * is first increasing and then decreasing.
 *                                                    
 * @param[in] n The length of an array.
 * @param[in] arr An array of numbers.
 *
 * @return Return the inverse count of a given array.
 */
long inverse_count(long n, long *arr) {
  // Array indexes. i starts from the left while j starts from the right.
  long i = 0;
  long j = n - 1;

  // A variable to store the amount of inversion.
  long inv = 0;
 
  // Iterate through the array and compare the left and right elements.
  for (long k = 0; k <= n - 1; k += 1) {
    if (arr[i] <= arr[j]) {
      inv += (n - j) - 1;
      i += 1;
    } else {
      j -= 1;
    }
  }

  // Every element after the peak is inverted.
  for (long k = 1; k <= (n - 1) - j - 1; k += 1) {
    inv += k;
  }

  return inv;
}

int main() {
  // Read the length of an array.
  long n = cs1010_read_long();

  // Read an array of numbers and check for memory allocation error.
  long *arr = cs1010_read_long_array(n);
  if (arr == NULL) {
    return -1;
  }

  // Print the amount of inversion in a given array.
  cs1010_println_long(inverse_count(n, arr));

  // Release the memory allocated for the array.
  free(arr);
  return 0;
}
