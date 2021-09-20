/**
 * CS1010 Semester 1 AY20/21
 * Assignment 1: Digits
 *
 * Read in a positive integer from the standard input and print
 * the sum of the square of each digit in the integer to the standard output.
 *
 * @file: digits.c
 * @author: Richard Willie (Group C08)
 */

#include "cs1010.h"

long square(long num)
{
  return num * num;
}

long sum_of_digits_square(long num)
{
  if (num / 10 == 0)
  {
    return square(num);
  }
  return square(num % 10) + sum_of_digits_square(num / 10);
}

int main()
{
  long num = cs1010_read_long();
  
  cs1010_println_long(sum_of_digits_square(num));
  return 0;
}
