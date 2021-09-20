/**
 * CS1010 Semester 1 AY20/21
 * Assignment 7: Sort
 *
 * This program sort an array of numbers. We assumed the
 * array to be first increasing and then decreasing. In
 * addition, all numbers are distinct.
 *
 * @file: sort.c
 * @author: Richard Willie (Group C08)
 */
#include "cs1010.h"

/**
 * Sort an array of numbers and print the array. The array
 * is first increasing and then decreasing.
 *
 * @param[in] n The length of an array.
 * @param[in] arr An array of numbers.
 * @param[in,out] new_arr An array to store the sorted array.
 */
void sort(long n, long *arr, long *new_arr) {
  // Array indexes. i starts from the left while j starts from the right.
  long i = 0;
  long j = n - 1;
  
  // Iterate through the array and compare left and right elements.
  for (long k = 0; k <= n - 1; k += 1) {
    if (arr[i] <= arr[j]) {
      new_arr[k] = arr[i];
      i += 1;
    } else {
      new_arr[k] = arr[j];
      j -= 1;
    }
  }
  
  // Print the sorted array.
  for (long k = 0; k <= n - 1; k += 1) {
    cs1010_print_long(new_arr[k]);
    cs1010_print_string(" ");
  }
  cs1010_println_string("");
}

int main() {
  // Read the length of an array.
  long n = cs1010_read_long();

  // Read an array of numbers and check for memory allocation error.
  long *arr = cs1010_read_long_array(n);
  if (arr == NULL) {
    return -1;
  }
  
  // Initialize a new array and check for memory allocation error.
  long *new_arr;
  new_arr = malloc((size_t)n * sizeof(long));
  if (new_arr == NULL) {
    return -1;
  }

  // Sort and print the array.
  sort(n, arr, new_arr);

  // Release the memory allocated for both arrays. 
  free(arr);
  free(new_arr);
  return 0;
}
