Question 1: Collatz (5 marks)
-----------------------------

The Collatz Conjecture was introduced by the mathematician
Lothar Collatz in 1937. Also known as the 3n+1 conjecture,
the problem can be stated very simply but yet no one can
prove that it is true or false. The conjecture states the
following:

Consider the following operation on a positive integer n: if
n is even, divide it by two; otherwise, triple it and add
one. Suppose we form a sequence of numbers by performing
this operation repeatedly, beginning with any positive
integer, then this process will eventually reach the number
1, for any initial positive integer n.

For instance, if n=10, then we have the sequence

10 -> 5 -> 16 -> 8 -> 4 -> 2 -> 1

The smallest number of steps taken by this process for n
to reach 1 is called the total stopping time. In the
example above, the total stopping time for 10 is 6.

Write a program collatz.c that reads in a positive integer
n from the standard input and finds out, among the numbers
between 1 to n, inclusive, which one has the largest total
stopping time. If two numbers have the same total stopping
time, we break ties by choosing the larger number as the
answer.

Your program should print to the standard output, the number
with the largest total stopping time and its corresponding
total stopping time in two different lines.

Sample Run
----------

ooiwt@pe114:~/as02-skeleton$ ./collatz
1
1
0
ooiwt@pe114:~/as02-skeleton$ ./collatz
10
9
19
ooiwt@pe114:~/as02-skeleton$ ./collatz
1000
871
178


Question 2: Triangle (4 marks)
------------------------------

Write a program `triangle.c` that draws an isosceles
triangle using ` ` (white space) and `#`.  The program must
read in a positive integer representing the height h of a
triangle.  The triangle must have exactly h rows.  Each row
must have exactly 2h-1 characters (including white spaces
but excluding a new line).  On each row, the sequence of "#"
characters must be centralized, padded by white spaces on
both sides.

Sample Run
----------

ooiwt@pe112:~/as02-ooiwt$ ./triangle
5
    #
   ###
  #####
 #######
#########
ooiwt@pe112:~/as02-ooiwt$ ./triangle
3
  #
 ###
#####
ooiwt@pe112:~/as02-ooiwt$ ./triangle
1
#

Hints
-----

1.
First, find the pattern to draw the triangle.

On row k, how many #s should you draw?  how many white
spaces should be padded on the left and the right?

2.
Write a function that draws a particular row of the triangle.
Then, call this function repeatedly in a loop.

Question 3: Prime (6 marks)
---------------------------

Write a program called `prime` that reads a positive integer
n from the standard input and prints the largest prime
smaller or equal to n.

Recall that a prime number is a number that is only
divisible by 1 and itself.

Your program must not make unnecessary checks or do
repetitive work.  In particular, once you found evidence
that a number is not a prime, there is no need to continue
checking.

Sample Run
----------
ooiwt@pe113:~/ex03-ooiwt$ ./prime
2
2
ooiwt@pe113:~/ex03-ooiwt$ ./prime
10
7
ooiwt@pe113:~/ex03-ooiwt$ ./prime
9223372036854775807
9223372036854775783

Hint
----
Write a boolean function `is_prime` to check if a given
number is prime.  Then call this function in a loop.


Question 4: Pattern (15 marks)
------------------------------

Even though the sequence of prime numbers appears to be
random, mathematicians have found some intriguing patterns
related to prime numbers. In this question, you are asked to
write a program to draw a variation of the "Parallax
Compression" pattern discovered by a software engineer,
Shaun Gilchrist.

The pattern visualizes the distribution of prime numbers in
a triangle, in the following way. The inputs given are an
interval n (n >= 1) and the height of the triangle h.

The triangle has h rows. The first row of the triangle has
one cell, the second row has three cells, the third row has
five, etc. The cells are centrally aligned so that visually
they form an equilateral triangle. We call the left-most
cell of each row the leading cell.

Each cell in the triangle contains n integers. The first
cell in the first row contains the numbers 1, 2, ..., n. The
leading cell of the next row, Row 2, contains n numbers
between n + 1 and 3n, with an increment of 2: i.e., n + 1, n
+ 3, n + 5, .., n + (2n - 1). The leading cell of the next
row, Row 3, contains the numbers 3n + 1 and 6n, with
increment of 3: i.e., 3n + 1, 3n + 4,3n + 7,..  3n+(3n−2),
etc.

For instance, if n is 5, the leading cells of the first
three rows contain the numbers
- {1, 2, 3, 4, 5},
- {6, 8, 10, 12, 14},
- {16, 19, 22, 25, 28},
respectively.

The rest of the cells in each row contains n numbers where
each is one more than a number contained in the cell on its
left. So, in Row 2, the numbers in the three cells are
- {6, 8, 10, 12, 14},
- {7, 9, 11, 13, 15}, and
- {8, 10, 12, 14, 16}.

In Row 3, the cells contain
- {16, 19, 22, 25, 28},
- {17, 20, 23, 26, 29},
- {18, 21, 24, 27, 30},
- {19, 22, 25, 28, 31}, and
- {20, 23, 26, 29, 32}.

Now, to visualize the distribution of primes, we do the
following, for each cell of the triangle that contains
either 1 or at least one prime, we print # to the standard
output at the corresponding position. Otherwise, we print (a
white space).

Your output must contain exactly h rows, each row exactly
2h−1 characters (including the white spaces but excluding
the newline). Note that in the sample runs below, the white
spaces are not visible.

Example
-------

ooiwt@pe114:~/as02-skeleton$ ./pattern
2 4
   #   
  # #  
 ## ## 
# # # #

To understand the output, consider the two numbers contained
in each cell, in the four rows:

Row 1: {1,2}
Row 2: {3,5} {4,6} {5,7}
Row 3: {7,10} {8,11} {9,12} {10,13} {11,14}
Row 4: {13,17} {14,18} {15,19} {16,20} {17,21} {18,22} {19,23}

Now, we check whether the numbers contained in each cell has
at least one prime, and replace them with either '#' or ' '.

Row 1: '#'
Row 2: '#', ' ', '#'
Row 3: '#', '#', ' ', '#', '#'
Row 4: '#', ' ', '#', ' ', '#', ' ', '#'

Drawing it nicely as an equilateral triangle yields the output

   #   
  # #  
 ## ## 
# # # #

Sample Runs
-----------

ooiwt@pe114:~/as02-skeleton$ ./pattern
11 11
          #          
          #          
        ## ##        
       # # # #       
      #### ####      
      # #   # #      
    ###### # ####    
   # # # # # # # #   
  ## ## ## ## ## ##  
  # # # #   # # # #  
###### ### ##########
ooiwt@pe114:~/as02-skeleton$ ./pattern
100 29
                            #                            
                           # #                           
                          ## ##                          
                         # # # #                         
                        #### ####                        
                       #   # #   #                       
                      ###### ######                      
                     # # # # # # # #                     
                    ## ## ## ## ## ##                    
                   # #   # # # #   # #                   
                  ########## ##########                  
                 #   # #   # #   # #   #                 
                ############ ############                
               # # #   # # # # # #   # # #               
              ## #  ##  # ## ## #  ##  # ##              
             # # # # # # # # # # # # # # # #             
            ################ ################            
           #   # #   # #   # #   # #   # #   #           
          ################## ##################          
         # #   # # # #   # # # #   # # # #   # #         
        ## ##  # ## #  ## ## ## ##  # ## #  ## ##        
       # # # # #   # # # # # # # # # #   # # # # #       
      ###################### ######################      
     #   # #   # #   # #   # #   # #   # #   # #   #     
    #### #### #### #### #### #### #### #### #### ####    
   # # # # # #   # # # # # # # # # # # #   # # # # # #   
  ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##  
 # # #   # # # # # #   # # # # # #   # # # # # #   # # # 
############################ ############################

Hints
-----

Solve this problem by breaking it down into smaller
problems.

In addition to drawing triangles and checking if a number
is prime, you might find the following sub-problems useful

- Find the first number of each leading cell of each row,
  given the row number and the interval n.

- Given the row, the col, and the interval n, does the
  cell contains a prime?
