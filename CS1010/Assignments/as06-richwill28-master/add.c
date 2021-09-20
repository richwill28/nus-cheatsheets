/**
 * CS1010 Semester 1 AY20/21
 * Assignment 6: Add
 *
 * This program adds two non-negative numbers which are
 * represented as strings from the standard inputs.
 *
 * @file: add.c
 * @author: Richard Willie (Group C08)
 */
#include "cs1010.h"
#include <stdbool.h>
#include <string.h>
#include <assert.h>

#define NUM(x) ((x) - '0')
#define CHAR(x) ((x) + '0')

/**
 * A function to convert two strings to numbers, then add said
 * number to get their sum. Next, print the sum.
 *
 * @param[in] long_num The longer number
 * @param[in] short_num The shorter number
 * @param[in,out] sum The sum of both numbers
 * @param[in] longer The length of the longer number
 * @param[in] shorter The length of the shorter number
 *
 * @pre longer >= shorter
 */
void add(char *long_num, char *short_num, char *sum, long longer, long shorter) {
  // To account for two digits sum larger than 9.
  long carry = 0;

  // Converts strings to numbers before adding them.
  for (long i = longer - 1; i >= 0; i -= 1) {
    if (i >= (longer - shorter)) {
      /*
       Check if the program is accessing the correct indexes
       within the memory allocated for short_num.
       */
      assert(i - (longer - shorter) >= 0);
      assert(i - (longer - shorter) <= shorter - 1);

      sum[i] = CHAR((NUM(long_num[i]) + NUM(short_num[i - (longer - shorter)]) + carry) % 10);
      carry = (NUM(long_num[i]) + NUM(short_num[i - (longer - shorter)]) + carry) / 10;
    } else {
      sum[i] = CHAR((NUM(long_num[i]) + carry) % 10);
      carry = (NUM(long_num[i]) + carry) / 10;
    }
  }

  // Print 1 if the sum has a leading 1 instead of 0.
  if (carry == 1) {
    cs1010_print_long(carry);
  }

  cs1010_print_string(sum);
  cs1010_println_string("");
}

int main() {
  char *n1 = cs1010_read_word();
  char *n2 = cs1010_read_word();
  
  // Read the length of the first and second strings.
  long len1 = (long)strlen(n1);
  long len2 = (long)strlen(n2);
  
  char *sum;

  /*
   If the length of the first string is longer, proceed to 'add'
   function. Otherwise, swap the parameter before calling the
   'add' function.
   */
  if (len1 >= len2) {
    sum = calloc((size_t)(len1), sizeof(char));
    if (sum == NULL) {
      return -1;
    }

    add(n1, n2, sum, len1, len2);
  } else {
    sum = calloc((size_t)(len2), sizeof(char));
    if (sum == NULL) {
      return -1;
    }

    add(n2, n1, sum, len2, len1);
  }

  free(n1);
  free(n2);
  free(sum);
  return 0;
}
