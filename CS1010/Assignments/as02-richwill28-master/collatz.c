/**
 * CS1010 Semester 1 AY20/21
 * Assignment 2: Collatz 
 *
 * Read in a positive integer n from the standard input,
 * then, for each number k between number 1 .. n, transform
 * it into 1 based on the steps below:
 * - if k is even, k = k/2
 * - else, k = 3k + 1
 * Find the number k that requires the most number of steps
 * between 1 and n, and print both k and the number of steps
 * to standard output. If there are multiple numbers with
 * the same number of steps, we prefer to output the larger
 * among these numbers.
 *
 * @file: collatz.c
 * @author: Richard Willie (Group C08)
 */
#include "cs1010.h"

void collatz(long n) {
  long largest_stop_time = 0;
  // largest total stopping time among the numbers
  
  long largest_number = 1;
  // number with the largest total stopping time

  for (long k = 1; k <= n; k += 1) {
    // repeat process from 1 to n

    long total_stop_time = 0;
    // total stopping time of k
    
    // the collatz sequence
    for (long i = k; i != 1; total_stop_time += 1) {
      // check whether i is even or odd
      if (i % 2 == 0) {
          i = i / 2;
        } else {
          i = 3 * i + 1;
        }
    }

    if (total_stop_time >= largest_stop_time) {
      largest_stop_time = total_stop_time;
      largest_number = k;
    }
  }

  cs1010_println_long(largest_number);
  cs1010_println_long(largest_stop_time);
}

int main() {
  collatz(cs1010_read_long());
  return 0;
}
