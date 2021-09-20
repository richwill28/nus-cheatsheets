/**
 * CS1010 Semester 1 AY20/21
 * Assignment 2: Pattern
 *
 * Read in two positive integers: an interval n (n >= 1) 
 * and the height of the triangle h.  Then draw a triangle
 * according to a given pattern.  
 *
 * @file: pattern.c
 * @author: Richard Willie (Group C08)
 */
#include "cs1010.h"
#include <math.h>
#include <stdbool.h>

long first_number(long n, long row) {
  return 1 + n * (row - 1) + n * ((row - 1) * (row - 2)) / 2;
  // return the first number of each rows
}

bool is_prime(long first_cell) {
  if ((first_cell == 1) || ((first_cell != 2) && (first_cell % 2 == 0))) {
    return false;
  }
  // 1 and all even numbers except 2 are not prime numbers

  double range = sqrt(first_cell);
  // iterating to sqrt(number) is sufficient in checking whether said number is prime or not

  for (long i = 2; i <= range; i += 1) {
    if (first_cell % i == 0) {
      return false;
    }
  }

  return true;
}

void print_row(long n, long h, long row) {
  long a = ((2 * h - 1) - (2 * row - 1)) / 2;
  // 'a' is the number of spaces on the left, which also equals to the number of spaces on the right

  for (long b = 1; b <= a; b += 1) {
    cs1010_print_string(" ");
  }
  // print spaces on the left

  long first_cell = first_number(n, row);
  // the first number of each cells

  for (long c = 1; c <= 2 * row - 1; c += 1) {
    long hash = 0;
    // a variable to check how many prime numbers present in a single cell

    for (long d = 1; d <= n; d += 1) {

      if ((first_cell == 1) || (is_prime(first_cell))) {
        hash += 1;
      }
      first_cell += row;
    }

    if (hash == 0) {
      // there are no prime numbers in the cell
      cs1010_print_string(" ");
    } else {
      // there are prime numbers in the cell
      cs1010_print_string("#");
    }

    first_cell = first_number(n, row) + c; 
  }

  for (long d = 1; d <= a; d += 1) {
    cs1010_print_string(" ");
  }
  // print spaces on the right

}

void triangle(long n, long h) {
  for (long row = 1; row <= h; row += 1) {
    print_row(n, h, row);

    cs1010_println_string("");
    // print new line at the end of each row
  }
}

int main() {
  triangle(cs1010_read_long(), cs1010_read_long());
  return 0;
}
