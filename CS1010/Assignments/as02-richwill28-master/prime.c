/**
 * CS1010 Semester 1 AY20/21
 * Assignment 2: Prime
 *
 * Reads in an integer n from standard input and print 
 * the largest prime smaller or equal to n.
 *
 * @file: prime.c
 * @author: Richard Willie (Group C08)
 */
#include "cs1010.h"
#include <math.h>
#include <stdbool.h>

bool is_prime(long k) {
  if ((k == 1) || ((k != 2) && (k % 2 == 0))) {
    return false;
  }
  // 1 and all even numbers except 2 are not prime numbers

  double range = sqrt(k);
  // iterating to sqrt(number) is sufficient in checking whether said number is prime or not

  for (long i = 2; i <= range; i += 1) {
    if (k % i == 0) {
      return false;
    }
  }

  return true;
}


long largest_prime(long num) {
  for (long k = num; k >= 2; k -= 1) {
    if (is_prime(k)) {
      return k;
    }
    // repeat process until a prime number is found
  }

  return 2;
  // assuming the minimum input is 2
}

int main() {
  cs1010_println_long(largest_prime(cs1010_read_long()));
  return 0;
}
