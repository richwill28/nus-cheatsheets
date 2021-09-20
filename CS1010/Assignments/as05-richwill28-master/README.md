Question 1: Contact (10 marks)
------------------------------

Contact tracing is an important step to contain the
on-going pandemic.  Suppose we are given the information
on who is in contact with who.  We wish to answer the
following question:

- Given two people A and B, have they been directly in
  contact with each other?

- Given two people A and B, is there a common person that
  they have been in contact with?  In other words, is there
  a third person C, where (i) A and C have been in contact
  with each other, and (ii) B and C have been in contact
  with each other?

We assume that "contact" is a symmetric relation.  If A has
been in contact with B, then B has been in contact with
A too.  Because of this, we can represent the contact
traces between n people as a lower triangular matrix
(_using jagged 2D array_).  A proper type to store in
each element of the matrix is `bool`.  To simplify our
life, however, we store each element of the matrix as a
`char`, with `'1'` representing a friendship connection,
`'0'` otherwise. The contact traces for n people is thus an
array of n strings, each string containing characters of
`'0'` and `'1'` only.  The first row of the matrix is a
string of length one; the second row is of length two;
third row, length three, etc.  The last character of
each string (i.e., the diagonal of the matrix) is `1`
since everyone has contact with him/herself.

For instance, suppose we have the following contact traces.
We represent each person with id 0, 1, 2, ...  Person with
id i has its information stored in Row i and Column i.
Recall that if Row i and Column j is 1, it means that
Person i has been in contact with Person j.
```
1
01
011
```

The contact traces above indicates that Person 1 and 2
have been in contact with each other.  Person 0 has not
been in contact with either.

As another example, the contact traces below shows an
extra person, Person 4.
```
1
01
011
1011
```

Person 4 has been in contact with both Person 1 and
Person 3.

Write a program `contact`, that reads from standard input:

- a positive integer n,
- followed by n lines of strings consisting of `'1'` or
  `'0'` representing the social network of these n people.
- followed by two positive integers m and n, representing
  the ids of a pair of people we are interested in querying.
  An input of i corresponds to the person whose contact
  information is stored in Row i and Column i.

Print, to the standard output, the following information:

- `direct contact` if there is direct contact between m
  and n
- `contact through x` if there is no direct contact but 
  there is an indirect contact between m and n through a 
  third person x (replace x with the actual id).  If there
  are multiple such third person, output the one with the
  smallest id.
- `no contact` if there is no direct nor indirect contact
  through a third person in the contact traces.

The purpose of this question is for you to practice
using a jagged 2D array.  Hence, _you are not allowed to
store the input matrix or intermediate matrices using a
rectangular array_, or you risk being penalized heavily
for this question.

### Sample Run
```
ooiwt@pe119:~/as05-skeleton$ cat inputs/contact.1.in
3
1
11
011
0 1
ooiwt@pe119:~/as05-skeleton$ ./social < inputs/contact.1.in
direct contact
ooiwt@pe119:~/as05-skeleton$ cat inputs/contact.2.in
3
1
11
011
0 2
ooiwt@pe119:~/as05-skeleton$ ./social < inputs/contact.2.in
contact through 1
ooiwt@pe119:~/as05-skeleton$ cat inputs/contact.3.in
4
1
01
011
1011
0 1
ooiwt@pe119:~/as05-skeleton$ ./contact < inputs/contact.3.in
no contact
```

Question 2: Social (15 marks)
------------------------------

You should solve this question after solving Question 1.
The functions you wrote for Question 1 might be useful
for solving this question.

The idea of six degrees of separation states that everyone
in the world is connected to every other person by at most
6 hops. Suppose that a person A is a friend of a person
B, then we say that A is one hop away from B. Consider a
friend of B, say C, who is not a friend of A. So C is a
friend of a friend of A. We say that C is two hops away
from A.  Six degrees of separation generally means there
is a chain of friendship that can connect any two people
in the world with no more than 6 hops.

In this question, we are going to compute the chain of
friendships up to the k degree. Suppose there are n people,
and we know the social network of these n people -- i.e.,
we know who is a friend with who. Write a program `social`
to compute a social network representing who is connected
to who via a friendship chain of degree k.

Similar to Question 1, we assume that friendship is
bi-directional -- if A is a friend of B, then B is a
friend of A.  We represent a social network as a lower
triangular matrix (using jagged 2D array) in the same
format as Question 1, where a `1` in Row i and Column j
means Person i and Person j are friends; `0` otherwise.

The social network below shows the friendship relations
between four people.

```
1
01
011
1011
```

Suppose now we consider the social network of degree
2. Person 0 is two hops away from Person 2 (Person 0 knows
Person 3, and Person 3 knows Person 2). Person 1 is also
two hops away from Person 3 (Person 1 knows Person 2 and
Person 2 knows Person 3).

The social network of degree 2 becomes:
```
1
01
111
1111
```

Write a program `social`, that reads from standard input
two positive integers n and k, followed by n lines
of strings consisting of '1' or '0' representing the
social network of these n people.  Print, to the standard
output, the social network formed by a friendship chain
of degree k.

The purpose of this question is for you to practice
using a jagged 2D array. Hence, you are not allowed to
store the input matrix or intermediate matrices using a
rectangular array, or you risk being penalized heavily
for this question.

### Hint
There are many ways to solve this problem.  The most
straightforward way is to compute the social network formed
by a friendship chain of degree i, N(i), changing i from
1 to k.

- N(1) is just the input;
- N(2) can be computed similar to how you solved Question 1.
- Now, given N(i-1) and N(1), can you compute N(i)?  

### Sample Run
--------------
```
ooiwt@pe119:~/as05-skeleton$ cat inputs/social.1.in
3 1
1
11
011
ooiwt@pe119:~/as05-skeleton$ ./social < inputs/social.1.in
1
11
011
ooiwt@pe119:~/as05-skeleton$ cat inputs/social.2.in
4 2
1
01
011
1011
ooiwt@pe119:~/as05-skeleton$ ./social < inputs/social.2.in
1
01
111
1111
ooiwt@pe119:~/as05-skeleton$ cat inputs/social.3.in
5 2
1
11
011
0011
10011
ooiwt@pe119:~/as05-skeleton$ ./social < inputs/social.3.in
1
11
111
1111
11111
```

## Question 3: Life (15 marks)
------------------------------

This question is another tribute to John H. Conway,
who passed away in April this year due to COVID-19.
The Game-of-Life is among his famous mathematical endeavors
(in addition to the Look-n-Say sequence).

The "Game of Life" is a game played on a two-dimensional
orthogonal grid of square cells, while each cell has
only two possible states: alive or dead. The game is
played in iterations. During each iteration, each cell
becomes alive or dead, depending on the state of its four
neighboring cells in the previous iteration. Interesting
patterns and moving behavior can be created, sometimes
infinitely, from an initial state. Refer to [wiki
page](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life)
for more details if you are interested.

In this problem, we are going to simulate a game of life
for a certain number of iterations, given a starting
state. Here is a complete description of the rules of
the simulation.

1. The universe is a bounded plane and can be simply
   referred to as a two-dimensional orthogonal grid of square
   cells with n rows and m columns. For convenience, we let
   row indexes as 0 to n-1 from top to bottom, and column
   indexes as 0 to m-1 from left to right. So in total,
   there are n X m cells.

2. The neighbor of a cell is defined as the eight cells
   that are either horizontally, vertically, or diagonally
   connected to the cell.

3. An initial state is given, with each cell is marked as
   either "live" or "dead".

4. In each iteration, a cell may switch its state,
   according to rules below, by referring to the state of
   the previous iteration:

* Any live cell with fewer than two live neighbors becomes dead
* Any live cell with two or three live neighbors remains alive.
* Any live cell with more than three live neighbors becomes dead.
* Any dead cell with exactly three live neighbors becomes alive
* Border cells, i.e., cells with row number 0 or $n-1$, or column 
  number 0 or $m-1$, are always dead. This is to simplify and 
  bound the universe.

Write a program `life` that reads, from the standard
inputs, three positive integers n (n > 2), m (m > 2) and
k, where n and m denote the number of rows and number of
columns of the universe (an n x m grid), and k is the
number of iterations to simulate.  It then reads, from
the standard input, n rows, with m characters in each row
representing the initial state.  Each character is either
alive (`#`) or dead (`.`).

The program then prints, to standard output, an animation
of the universe for k iterations.  The output should only
contain n rows with m characters in each row.  Similarly,
you must use `#` to represent a live cell, and `.` to
represents a dead cell.

### Animation on Screen

We have provided a few lines of code in the skeleton file.
You should insert this at appropriate places:

```
  cs1010_clear_screen();
  // TODO(by student) draw the universe
  usleep(250*1000);
```

Line 1 in the code above calls a function from the CS1010
I/O library which clears your screen and places the cursor
on the top left corner of the terminal.   Line 3 calls
the system function `usleep` that takes in the number
of microseconds.  Calling `usleep` causes the program
to pause for that amount of time.  We set the sleeping
time to 250ms.  You can reduce if you wish but you must
not increase this beyond 250ms or your program might fail
when we test it during grading.

### Sample Run

- The pattern from `life.1.in` is called a blinker.
- The pattern from `life.2.in` is called a pentadecathlon.
- The pattern from `life.4.in` is called a pulsar.

We provide a total of seven patterns for you to play with.
You can run
```
~cs1010/life < inputs/life.1.in
```
for instance, to observe the output from the program.

### Testing and Debugging

Note that `make` will not automatically test `life`
for you.  Before you submit, you should check your output
is correct, by running
```
./test.sh life
```
manually.  If you do so, you may wish to reduce the
sleeping time or do not call `usleep` altogether to speed
up testing.  Sleeping time only affects the animation,
not the correctness of the output.

To help you debug, you can redirect the output from life
into a file, and use `vim -d` to view the differences.
```
ooiwt@pe119:~/as05-skeleton$ ./life < inputs/life.3.in > OUT
ooiwt@pe119:~/as05-skeleton$ vim -d OUT outputs/life.3.out
```

Students interested in playing more games of life can also
check out LifeWiki (http://conwaylife.com/wiki/Main_Page)
for more patterns and spiral-click into various information
about this game.
