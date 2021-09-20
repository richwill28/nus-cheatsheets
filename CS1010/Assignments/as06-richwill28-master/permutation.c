/**
 * CS1010 Semester 1 AY20/21
 * Assignment 6: Permutation
 * 
 * Given two strings of lowercase letters 'a' to 'z', with 
 * first string * no longer than the second. This program 
 * checks if the 1st string is a permutation of a substring 
 * of the 2nd one.
 *
 * @file: permutation.c
 * @author: Richard Willie (Group C08)
 */
#include "cs1010.h"
#include <stdbool.h>
#include <string.h>

/**
 * Count the frequencies of each alphabet in the first word. Next,
 * count the frequencies of each alphabet in the consecutive substring
 * of the second word. Return true if the first word and the substring
 * of the second word are permutation of each other, otherwise return
 * false.
 * 
 * @param[in] s1 The first word
 * @param[in] s2 The second word
 * @param[in,out] c1 The frequencies of each alphabet in the first word
 * @param[in,out] c2 The frequencies of each alphabet in the substring
 *                   of the second word
 * @param[in] k The length of the first word
 * @param[in] n The length of the second word
 *
 * @return Return true or false
 */
bool perm(char *s1, char *s2, long *c1, long *c2, long k, long n) {
  /*
   Initialize a flag to check whether the frequencies of each alphabet
   from both words are equal.
   */
  long flag = 0;

  // Count the frequencies of each alphabet in the first word.
  for (long i = 0; i <= k - 1; i += 1) {
    c1[s1[i] - 'a'] += 1;
  }

  for (long i = 0; i <= n - 1; i += 1) {
    // Count the frequencies of the alphabets in the second word.
    c2[s2[i] - 'a'] += 1;

    // Compare both frequencies for each consecutive substring of length k.
    if (i >= k - 1) {
      for (long j = 0; j <= 25; j += 1) {
        if (c1[j] - c2[j] == 0) {
          flag += 1;
        }
      }

      // If flag is 26, then the strings are permutation of each other.
      if (flag == 26) {
        return true;
      }
      flag = 0;

      // Delete the count of previous substring to prevent overcounting.
      c2[s2[i - (k - 1)] - 'a'] -= 1;
    }
  }

  return false;
}

int main() {
  char *s1 = cs1010_read_word();
  char *s2 = cs1010_read_word();

  // Initialize arrays indicating the frequencies of each alphabet.
  long c1[26] = {0};
  long c2[26] = {0};

  // Calculate the length of the first and second strings.
  long k = (long)strlen(s1);
  long n = (long)strlen(s2);

  if (perm(s1, s2, c1, c2, k, n)) {
    cs1010_println_string("YES");
  } else {
    cs1010_println_string("NO");
  }

  free(s1);
  free(s2);
  return 0;
}
