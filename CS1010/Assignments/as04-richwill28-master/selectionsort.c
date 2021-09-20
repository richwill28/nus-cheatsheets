/**
 * CS1010 Semester 1 AY18/19
 * Assignment 4: SelectionSort
 *
 * Sort an array of number in increasing order. Firstly, find
 * the largest number in the array. Secondly, swap its position
 * with the last number in the array. Thirdly, print the new
 * array. Lastly, repeat the algorithm with the second, third,
 * fourth, etc. largest number and last number instead.
 *
 * @file: selectionsort.c
 * @author: Richard Willie (Group C08)
 */
#include "cs1010.h"

/**
 * Find the maximum number from a given array and return its
 * position/index in the array.
 *
 * @param[in] n The length of the array.
 * @param[in] list The array of number(s).
 * 
 * @return Return the position/index of the maximum number in
 *         the array.
 */
long max_position(long n, const long list[]) {
  // Initialize the position of the maximum number so far.
  long max_so_far = 0;

  /*
   Find the position of the maximum number by iterating through
   the array.
   */
  for (long i = 1; i <= n - 1; i += 1) {
    if (list[i] >= list[max_so_far]) {
      max_so_far = i;
    }
  }

  return max_so_far;
}

/**
 * Read the length of an array and the array of numbers. Then,
 * swap the maximum number with the last number in the array
 * and print the new array.
 *
 * @param[in] n The length of the array.
 * @param[in,out] list The array of numbers.
 */
void selectionsort(long n, long list[]) {
  // Repeat the algorithm until the array is sorted.
  for (long i = n; i >= 2; i -= 1) {
    // Initialize the position of the maximum number.
    long position = max_position(i, list);

    // Initialize a new variable to store the maximum number.
    long max = list[position];
    
    // Move the last number to the position of the maximum number.
    list[position] = list[i - 1];

    // Move the maximum number to the position of the last number.
    list[i - 1] = max;

    // Print the new array.
    for (long j = 0; j <= n - 1; j += 1) {
      cs1010_print_long(list[j]);
      cs1010_print_string(" ");
    }
    cs1010_println_string("");
  }
}

int main() {
  // Read the length of an array.
  long n = cs1010_read_long();

  // Read an array of numbers.
  long *list = cs1010_read_long_array(n);

  // Return -1 if there is a memory allocation error.
  if (list == NULL) {
    return -1;
  }

  // Run the selection sort algorithm.
  selectionsort(n, list);
  
  // Release the memory allocated for 'list'.
  free(list);

  return 0;
}
