/**
 * CS1010 Semester 1 AY20/21
 * Assignment 1: Suffix
 *
 * Read in a number and prints out the number with the appropriate suffix.
 *
 * @file: suffix.c
 * @author: Richard Willie (Group C08)
 */
#include "cs1010.h"

void print_with_suffix(long n)
{
  if ((n % 10 == 1) && (n % 100 != 11))
  {
    cs1010_print_long(n);
    cs1010_println_string("st");
  }
  else if ((n % 10 == 2) && (n % 100 != 12))
  { 
    cs1010_print_long(n);
    cs1010_println_string("nd");
  }
  else if ((n % 10 == 3) && (n % 100 != 13))
  {
    cs1010_print_long(n);
    cs1010_println_string("rd");
  }
  else
  {
    cs1010_print_long(n);
    cs1010_println_string("th");
  }
}

int main() 
{
  long num = cs1010_read_long();
  print_with_suffix(num);

  return 0;
}
