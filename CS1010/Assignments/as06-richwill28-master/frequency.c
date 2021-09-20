/**
 * CS1010 Semester 1 AY20/21
 * Assignment 6: Frequency
 * 
 * Given 2 equal length string of small letter 'a'
 * to 'z', this program checks if both have the
 * same set of alphabets appearing the same number
 * of time.
 *
 * @file: frequency.c
 * @author: Richard Willie (Group C08)
 */
#include "cs1010.h"
#include <stdbool.h>
#include <string.h>

/**
 * +1 frequency for each alphabet encountered in the first word,
 * then -1 frequency for each alphabet encounter in the second word.
 * If the resulting frequencies of each aplhabet is 0, then the words
 * are permutation of each other.
 *
 * @param[in] s1 The first word
 * @param[in] s2 The second word
 * @param[in,out] count The frequencies of each alphabet in a word
 * @param[in] len The length of the first word
 *
 * @return Return true or false
 */
bool frequency(char *s1, char *s2, long *count, long len) {
  for (long i = 0; i <= len - 1; i += 1) {
    count[s1[i] - 'a'] += 1;
    count[s2[i] - 'a'] -= 1;
  }

  for (long i = 0; i <= 25; i += 1) {
    if (count[i] != 0) {
      return false;
    }
  }

  return true;
}

int main() {
  char *s1 = cs1010_read_word();
  char *s2 = cs1010_read_word();

  // Initialize an array indicating the frequencies of each alphabet.
  long count[26] = {0};

  // The length of the first string.
  long len = (long)strlen(s1);

  if (frequency(s1, s2, count, len)) {
    cs1010_println_string("YES");
  } else {
    cs1010_println_string("NO");
  }

  free(s1);
  free(s2);
  return 0;
}
