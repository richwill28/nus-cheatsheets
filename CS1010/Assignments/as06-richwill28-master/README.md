Assignment 6: Add, Frequency, Permutation
-----------------------------------------

Question 1: Add (15 marks)
--------------------------

In this question, you are asked to write a program that adds
two non-negative numbers which can be arbitrarily large.
The types provided by C can only represent a number up to a
certain value.  We have seen that `long long int` is not
even big enough to represent 21!.

For this question, we will represent a number using an
arbitrarily long string consisting of characters (of type
`char`) `'0'` to `'9'` (note: not integer 0 to 9).  C
supports arithmetic operations on `char` values as well.  To
convert between the numerical value of a digit character, we
can do the following:

- To convert from a digit character to its numerical value,
  we subtract the `char` `'0'`.  For instance, `'6' - '0'`
  will give us the value `6`.

- To convert from a numerical value of a digit to its
  character, we add the `char` `'0'`.  For instance, `6 + '0'`
  will give us the character `'6'`.

You can read a sequence of non-space characters from the
standard input using `cs1010_read_word`, and print a
sequence of characters (i.e., a string) to the standard
output using `cs1010_println_string`.

Write a program `add` that reads from the standard input two
non-negative numbers represented as strings consisting of
digits '0' to '9', and prints to the standard output the sum
of the two numbers.

Your program should use `assert` liberally to check if you
are accessing the correct indices.  Every time you access an
array element with index `i`, you should check if `i` is
within the memory allocated for the array.

You will likely need to use the C standard library function
`strlen`, which returns you the number of characters in a
string (excluding the terminating '\0').  Look up on how to
use this function on your own.  You might need casting of
the return value to avoid warnings.

The grading criteria for this question is:

|             | Marks|
|-------------|------|
|Documentation| 2    |
|Correctness  | 11   |
|Efficiency   | 2    |

Your algorithm must add in O(n) time to receive 2 efficiency marks.

### Sample Run
```
ooiwt@pe119:~/as06-skeleton$ ./add
1
1
2
ooiwt@pe119:~/as06-skeleton$ ./add
7
8
15
ooiwt@pe119:~/as06-skeleton$ ./add
999999
1
1000000
ooiwt@pe119:~/as04-skeleton$ ./add
1400060514000605140006051400605
19330911933091193309119330911933091
19332311993605193914259336963333696
```

# Question 2: Frequency (10 marks)
----------------------------------

Write a program `frequency`, that, given two strings,
consists of alphabets 'a' to 'z', checks if the same
alphabet appears the same number of times in both strings
(but maybe in a different order).

For instance, `nus` and `sun` have the same alphabets
appearing the same number of times.  But `soc` and `nus`
do not.

Your program should read, from the standard input,

- a string S1, consists of n lowercase alphabets (`a` to `z`)
- a string S2, consists of n lowercase alphabets (`a` to `z`)

and print, to the standard output, `YES` if the same set of
alphabets appear the same number of times in both strings,
`NO` otherwise.

Your solution must take no more than O(n) time.  

The grading criteria for this question is:
|             | Marks|
|-------------|------|
|Documentation| 2    |
|Correctness  | 3    |
|Efficiency   | 5    |

A solution that takes longer than O(n) will receive 0 marks
for efficiency.  Furthermore, your solution needs to be
correct to receive marks for efficiency.

### Sample Run
```
ooiwt@pe119:~/as06-skeleton$ ./frequency
sun
nus
YES
ooiwt@pe119:~/as06-skeleton$ ./frequency
nus
soc
NO
```

# Question 3: Permutation (15 marks)
------------------------------------

Write a program `permutation`, that, given two strings,
consists of alphabets 'a' to 'z', S1 and S2, checks if S2 is
a permutation of some substring of S1.  A substring of
length k is a consecutive sequence of k characters from a
string.

For instance, `nus` is a permutation of a substring of
`suntec`, since `suntec` contains `sun`.  `ntu` is also a
permutation of a substring of `suntec`, since `suntec`
contains `unt`.  `smu` is not a permutation of any substring
of `suntec`.

Your program should read, from the standard input,

- a string S1, consists of k characters, chosen from `a` to `z`
- a string S2, consists of n characters, chosen from `a` to `z`

and print, to the standard output, `YES` if S2 is a
permutation of some substring of length k from S1, and `NO`
otherwise.

The grading criteria for this question is:

|             | Marks|
|-------------|------|
|Documentation| 2    |
|Correctness  | 5    |
|Efficiency   | 8    |  

Your solution needs to be correct to receive marks for
efficiency.  The efficiency marks are given as follows:

- $O(nk^2)$ solution: 2 marks
- $O(nk)$ solution: 4 marks
- $O(n+k)$ solution: 8 marks

Note that `test.sh` does not check for efficiency.

!!! hint
    Use your code from Question 2 and avoid duplicated work!

```
ooiwt@pe119:~/as06-skeleton$ ./permutation
suntec
nus
YES
ooiwt@pe119:~/as06-skeleton$ ./permutation
suntec
smu
NO
```
