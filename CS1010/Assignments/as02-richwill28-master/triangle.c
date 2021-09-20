#include "cs1010.h"
/**
 * CS1010 Semester 1 AY20/21
 * Assignment 2: Triangle
 *
 * Read in a positive integer h from standard input and 
 * print an equlateral triangle of height h.
 *
 * @file: triangle.c
 * @author: Richard Willie (Group C08)
 */

void row(long h, long n) {
  long j = ((2 * h - 1) - (2 * n - 1)) / 2;
  // j is the number of spaces on the left, which also equals to the number of spaces on the right
  
  for (long k = 1; k <= j; k += 1) {
    cs1010_print_string(" ");
  }
  // print j spaces on the left

  for (long l = 1; l <= 2 * n - 1; l += 1) {
    cs1010_print_string("#");
  }
  // print (2 * i - 1) #s on the middle

  for (long m = 1; m <= j; m += 1) {
    cs1010_print_string(" ");
  }
  // print j spaces on the right
}

void triangle(long h) {
  for (long n = 1; n <= h; n += 1) {
    row(h, n);
    // print nth row

    cs1010_println_string("");
    // print new line at the end of each row
  }
}

int main() {
  triangle(cs1010_read_long());
  return 0;
}
