/**
 * CS1010 Semester 1 AY20/21
 * Assignment 4: Mastermind
 *
 * This program simulates the game 'mastermind'. Firstly, it
 * reads a 4-letter code from the standard inputs. Then, it
 * reads in a sequences of guesses. For each guess, the program
 * prints the number of pegs that are correct in both position
 * and color as well as the number of pegs that are correct in
 * color but not position. However, the program does not double
 * count, so the maximum total of the two numbers is 4. Moreover,
 * unlike the original board game, the number of guesses in this
 * program is unlimited.
 *
 * @file: mastermind.c
 * @author: Richard Willie (Group C08)
 */
#include "cs1010.h"
#include <stdbool.h>

/**
 * The following function read the code and the guess from the
 * inputs, print the number of black and white pegs, and finally
 * return the number of black pegs.
 *
 * @param[in] code The secret code to be guessed. 
 * @param[in] guess The guess.
 *
 * @return Return the number of black pegs.
 *
 * @post (black + white) <= 4.
 */
long mastermind(const char code[], char guess[]) {
  // Initialize a new duplicate of the code.
  char new_code[5] = "....";
  for (long i = 0; i <= 3; i += 1) {
    new_code[i] = code[i];
  }
  
  // Initialize the number of black and white pegs.
  long black = 0;
  long white = 0;

  /*
   Compare each letter of the code to the guess. If both the
   color and position are correct, increment the number of
   black pegs and change the letters to '.' to prevent double
   counting.
   */
  for (long i = 0; i <= 3; i += 1) {
    if (new_code[i] == guess[i]) {
      black += 1;
      new_code[i] = '.';
      guess[i] = '.';
    }
  }

  /*
   Compare each letter of the code to the guess. Only if the
   color is correct, increment the number of white pegs and
   change the letters to '.' to prevent double counting.
   */
  for (long i = 0; i <= 3; i += 1) {
    if (new_code[i] != '.') {
      /*
       Initialize 'flag' to prevent error from checking excessively
       letters which have already satisfied the condition of white
       pegs before.
       */
      long flag = 0;

      for (long j = 0; j <= 3; j += 1) {
        if (flag == 0) {
          if (new_code[i] == guess[j]) {
            white += 1;
            flag += 1;
            guess[j] = '.';
         }
        }
      }
    }
  }
  
  // Print the number of black and white pegs
  cs1010_print_long(black);
  cs1010_print_string(" ");
  cs1010_println_long(white);

  return black;
}

int main() {
  // Read the code.
  char *code = cs1010_read_word();
  
  // Initialize the number of black pegs.
  long black;

  // Repeat until the guess is correct.
  do {
    // Read the guess.
    char *guess = cs1010_read_word();

    // Run the mastermind function and return its value to 'black'.
    black = mastermind(code, guess);

    // Release the memory allocated for 'guess'.
    free(guess);
  } while (black <= 3);

  // Release the memory allocated for 'code'.
  free(code);

  return 0;
}
