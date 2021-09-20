/**
 * CS1010 Semester 1 AY20/21
 * Assignment 3: ID
 *
 * Read in a number and print out the check code 
 * according to NUS student ID algorithm.
 *
 * @file: id.c
 * @author: Richard Willie (Group C08)
 */
#include "cs1010.h"

char id(long n) {
  long N = 0;

  // Sum the digits of n
  while (n != 0) {
    N += n % 10;
    n /= 10;
  }

  long R = N % 13;
  char code[13] = "YXWURNMLJHEAB";
  
  return code[R];
}

int main() {
  putchar(id(cs1010_read_long()));
  return 0;
}
