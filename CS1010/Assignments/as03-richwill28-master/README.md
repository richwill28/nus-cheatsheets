Question 1: ID (5 marks)
------------------------

Your NUS student id has a letter at the end. This letter is
called a check code and is a form of redundancy check used
for detecting errors, especially when your student id is
manually entered into a software application.

Your check code is calculated by:

1. Sum up the digits in your student id. Let the sum be N.
2. Divide N by 13, and take the remainder. Let the remainder
   be R.  
3. Look up the table below:

R  | Check Code
---|------------
0  | Y
1  | X
2  | W
3  | U
4  | R
5  | N
6  | M
7  | L
8  | J
9  | H
10 | E
11 | A
12 | B

Write a program `id.c` that reads in a positive integer
containing the digits of a student's id from the standard
input. Print out the check code to the standard output.

You can use the `putchar` function to print a single `char`
variable to the standard output.

Use array and loops to solve this problem instead of
conditional statements.  Solutions that use conditional
statements (in any form) will receive a 0.

Sample Run
----------
```
ooiwt@pe116:~/as03-ooiwt$ ./id
1933091
Y
ooiwt@pe116:~/as03-ooiwt$ ./id
3364497
E
ooiwt@pe116:~/as03-ooiwt$ ./id
1111111111111
Y
```

Question 2: Kendall (7 Marks)
-----------------------------

Suppose that we are given a set of items and we ask two
different parties to rank the items according to some order.
We may get two different orders of the items. How do we
measure how similar (or dissimilar) the two rankings are?

For example, consider a search engine that returns a list of
web pages ranked by their relevance to the search query. A
user may not always agree with the ranking of the search
engine and may judge the relevance of the search result
differently, i.e., the user may have his or her own ranking.
This measurement of similarity between the ranking by the
search engine and the ranking by the user gives us a metric
on how good the search engine result is. The more similar it
is to the ranking of the user, the better the search engine
is in ranking in the search results.

One way to measure the similarity of two rankings is the
Kendall tau distance. You will write a program `kendall` that
calculates the normalized Kendall tau distance for this
question.

We will represent a ranking by the order of the items. The
first item is ranked 1, the second is ranked 2, and so on. To
simplify the problem, we take one of the rankings that we
want to calculate the Kendall tau distance on, and label the
items sequentially, as the sequence 1, 2, 3, 4, 5, ... n,
where n is the number of items. We call this the base
ranking. The other ranking will then be a permutation of the
numbers 1 to n.

For example, suppose we have three items A, B, C. The first
ranking ranks the items as B, C, A. The second ranking ranks
the items C, A, B. After relabelling the first ranking as 1,
2, 3, the second ranking becomes 2, 3, 1.

The Kendall tau distance counts the number of pairs of items
in one ranking that are ranked in a different order in the other
ranking. In the example above, we have three possible pairs:

Pair | Ranking 1 | Ranking 2
-----|-----------|----------
A-B  | B then A  | A then B
A-C  | C then A  | C then A
B-C  | B then C  | C then B

Out of the three pairs, the pair A-B and B-C are ordered
differently in the two rankings, so that Kendall tau
distance is 2.

The normalized Kendall tau distance is the ratio of the
number of pairs ranked in a different order to all possible
number of pairs.

In the example above, the normalized Kendall distance is
2/3 = 0.6666.

Your program should read the following from the standard
input:

- The first positive integer, n, is the number of items
  (n > 1).  
- The next n numbers is a permutation of integers between
  1 to n. This corresponds to the ranking of the items
  from 1 to n.

Your program should print the normalized Kendall tau
distance between the ranking read above and the base ranking
(1, 2, 3, .. n) to the standard output.

Sample Run
----------
```
ooiwt@pe118:~/as03-skeleton$ ./kendall
3
2 3 1
0.6667
ooiwt@pe118:~/as03-skeleton$ ./kendall
10
1 2 3 4 5 6 7 8 9 10
0.0000
ooiwt@pe118:~/as03-skeleton$ ./kendall
6
6 5 4 3 2 1
1.0000
```

Question 3: Max (7 marks)
-------------------------

Write a program max that finds the maximum value from a list
of n integers L.

Instead of doing this with a loop, solve this question with
_recursion_. Write a function

```
long max(const long list[], long start, long end)
{
	:
}
```

that calls itself and return the maximum value among the
array elements list[start] .. list[end - 1]. It should split
the input list into two halves (roughly), find the maximum
of the left half, find the maximum of the right half, and
return the larger of these two maximums.

In the function definition above, the keyword `const` (short
for constant) is used to annotate that the array list is
meant to remain unchanged.

The program should read the following from the standard
inputs:

- The first number is a positive integer n.
- The next n numbers correspond to the list of integers L.

The program should then print to the largest value among
the inputs to the standard output.

Note that you are not allowed to use loops of any kind in
your solution (`for`, `while`, `do-while`).

Sample Run
----------

```
ooiwt@pe116:~/ex04-ooiwt$ cat input
5
-5 3 1 8 2
ooiwt@pe116:~/ex04-ooiwt$ ./max < input
8
```

Question 4: Counting Sort (11 Marks)
------------------------------------

Sorting is a fundamental computational problem: given a list
of items, we want to rearrange the items in some order.

In this question, you will write your first sorting
algorithm, called counting sort. This is an extremely fast
algorithm for sorting positive integers if the range of the
integers is limited.

The idea of counting sort is that, given the list of
integers (each guaranteed to be between 1 to k) to sort, we
count how many times 1 appear in the list, how many times 2
appears in the list, etc. Finally, we print out each number
between 1 to k, according to how many times they appear in
the list, skipping those numbers who do not appear.

For instance, suppose we have 6 integers between 1 to 9:

 5 5 3 2 8 2

We first count how many times each number appears. Then we
print the sorted list the following way: 2 appears twice, so
we print

2
2

The number 3 appears once, we print

3

The number 5 appears twice, we print

5
5

Finally, 8 appears once, we print

8

The sorted output is thus

2
2
3
5
5
8

which is the numbers sorted in increasing order.

Write a program countingsort.c that reads the following in
order from the standard input:

-  n, the number of integers to sort (n > 0).
-  The next n numbers are the integers to be sorted, each
   guaranteed to be between 0 and 10000.

Sort the integers using the algorithms above and print the
following:

- The number of times each number appears.  On each line,
  print two numbers, i and the number of times i appears
  in the input, separated by a space, in increasing order
  of i.  Skip any number that does not appear in the input.

- The numbers in the input, in increasing order.

Note that if you use any other algorithms to sort the
numbers, you will be penalized heavily.

Further, your code should go through every element in the 
array returned by the CS1010 I/O library exactly once.  
You will loose 1 mark for efficiency if you go through it 
more than once.

Sample Run
----------
```
ooiwt@pe118:~/as03-skeleton$ ./countingsort
6
5 5 3 2 8 2
2 2
3 1
5 2
8 1
2
2
3
5
5
8
ooiwt@pe118:~/as03-skeleton$ ./countingsort
3
256 872 112
112 1
256 1
872 1
112
256
872
```
