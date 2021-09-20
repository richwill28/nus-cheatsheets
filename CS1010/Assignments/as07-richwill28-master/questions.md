# Assignment 8: Peak, Sort, Inversion
----------------------------------------

## Question 1: Peak (10 marks)
------------------------------

Ash helped his professor, Professor Willow, to conduct a
topographic survey of a piece of land.   He walked in a
straight line, noting down the elevation of the land at
every centimeter. After he is done, he passed the data to
Professor Willow.  The professor asked, "what is the peak
elevation of the land?"  Ash did not know the answer!  He
could write a program to scan through the millions of data
points he collected, but he knew that since you have taken
CS1010, you can do a better job.  So, Ash asked for your help.

You first clarify the problem with Ash: "What is a peak?"
To which Ash explained that _a peak is a location that is
strictly higher than the surrounding locations._   Ash
explained further the pattern in the data: the elevation
always either remains the same or increases as he walks.
After he passed the peak, the elevation always either
remains the same, or decreases.  But he cannot remember if
he ever encountered a peak -- it might be possible that the
elevations data is always non-decreasing, or non-increasing,
or there is a flat plateau where there are multiple highest
locations with the same elevation.  So, a peak might not
exist.  But if there is a peak, it is guaranteed that there
is exactly one peak.

"Please, can you help me solve it in O(log n) time?"  Ash
pleaded.  "Piece of cake!"  You said.

Write a program `peak` that reads from the standard input
the following:

- An integer n (n >= 3), followed by
- n integers, each representing the elevation of a location
  surveyed by Ash

Then, prints, to the standard output, the index of the
location of the peak if it exists, or `no peak` if a peak
does not exist.   The first elevation has an index of 0.

An O(n) solution is trivial.  You will get 0 marks if your
solution simply scans through the array linearly looking for
an elevation that is the peak.  That is what Ash would do
anyway!  To get full marks for correctness and efficiency
for this solution, your code should run in O(log n) time
when the input elevations are all distinct.  Your code is
allowed to take longer than O(log n) if there are equal
elevations in the input data, with the extreme case of
O(n) when all elevation values are the same (the land is
completely flat).

Hint: Consider the middle element and its neighbors.  Can we
tell if the peak lies on the left or the right of the middle
element?

### Sample Run
```
ooiwt@pe119:~/as07-weitsang$ cat inputs/peak.1.in
5
1 3 5 4 2
ooiwt@pe119:~/as07-weitsang$ ./peak < inputs/peak.1.in
2
ooiwt@pe119:~/as07-weitsang$ cat inputs/peak.2.in
3
1 2 3
ooiwt@pe119:~/as07-weitsang$ ./peak < inputs/peak.2.in
no peak
ooiwt@pe119:~/as07-weitsang$ cat inputs/peak.3.in
4
-1 2 2 -1
ooiwt@pe119:~/as07-weitsang$ ./peak < inputs/peak.3.in
no peak
```

## Question 2: Sort (10 marks)
---------------------------------

Professor Willow is teaching a huge class at the university.
He finished grading a test and he passed a huge pile of test
scripts to Ask.  He asked Ash to help him enter the grades
into a spreadsheet, in increasing order of the student id.
Ash said, "Sure!  The scripts are sorted, right?"  The
professor answered, "Almost!  The top portion of the pile is
sorted in increasing order.  The rest, in decreasing order."
The professor then left abruptly, leaving Ash to wonder how
to deal with the test scripts.  Ash needed to sort the
scripts in increasing order of the student id.  So he
messaged you to help him figure out an efficient algorithm
to do this.  "No problemo!", you said, "Can be done in
O(n)!"  You said.  So you went ahead and wrote out the
following program to show Ash how he can solve his problem
in O(n) time.

Write a program `sort` that reads, from the standard input,
the following:

- An integer n (n >= 1), followed by
- n integers, each representing the student ids.

The student ids are unique, i.e., there is no duplicate in
the inputs.  The student ids from the inputs are arranged in
such a way that, the first k are in increasing order, the
remaining n - k are in decreasing order.  k is not given
in the input, and 0 <= k <= n.

Then, prints, to the standard output, the student ids in
increasing order.

An O(n^2) or O(n log n) solution is trivial.  You will
get 0 marks for correctness and efficiency if your solution
simply uses one of the existing sorting algorithms to sort
the scripts.  Note also that you cannot use counting sort
here since a student id can be represented with an
arbitrarily large integer (but still fit in a `long`).

Hint: Scan the input from both sides: front and back.

### Sample Run
```
ooiwt@pe119:~/as07-weitsang$ cat inputs/sort.1.in
5
1 3 5 4 2
ooiwt@pe119:~/as07-weitsang$ ./sort < inputs/sort.2.in
1 2 3 4 5
ooiwt@pe119:~/as07-weitsang$ cat inputs/sort.2.in
3
1 20 300
ooiwt@pe119:~/as07-weitsang$ ./sort < inputs/sort.2.in
1 20 300
ooiwt@pe119:~/as07-weitsang$ cat inputs/sort.3.in
1
-100
ooiwt@pe119:~/as07-weitsang$ ./sort < inputs/sort.3.in
-100
```

## Question 3: Inversion (15 marks)
-----------------------------------

Professor Willow called Ash in the evening.  "Mr. Ketchum, I
told you that the pile of scripts is almost sorted.  But I
do not know what it means by _almost_ actually.  Can you
help me figured out how to quantify that?"  Ash is clueless
as well.  So he called you.  "Ah, I learned this in CS1010
Assignment 3," you boasted, "In the problem Kendall, we
counted the number of inversions."  You then go on to
explain that an inversion is a pair of scripts that are out
of order.  A perfectly sorted pile of scripts would have
zero inversion, and an inversely sorted pile of scripts
would have n(n-1)/2 inversions.  "Let me help you
to do this, in O(n) time!"

Write a program `inversion` that reads, from the standard input,
the following:

- An integer n (n >= 1), followed by
- n integers, each representing the student ids.

The student ids are unique, i.e., there is no duplicate in
the inputs.  The student ids from the inputs are arranged in
such a way that, the first k are in increasing order, the
remaining n - k are in decreasing order.  k is not given
in the input, and 0 <= k <= n.

You program should then prints, to the standard output, the
number of inversions in the input.

You have already solved this problem in Assignment 3 in
O(n^2) time, so, an O(n^2) solution would receive 0
marks.  An O(n log n) solution will get 8 marks for
correctness and efficiency at most. (Hint for O(n log n)
solution: binary search).  To get full marks for correctness
and efficiency, you need to produce an O(n) solution (Hint
for O(n): sort).

### Sample Run
```
ooiwt@pe119:~/as07-weitsang$ cat inputs/inversion.1.in
5
1 3 5 4 2
ooiwt@pe119:~/as07-weitsang$ ./inversion < inputs/inversion.1.in
4
ooiwt@pe119:~/as07-weitsang$ cat inputs/inversion.2.in
3
1 20 300
ooiwt@pe119:~/as07-weitsang$ ./inversion < inputs/inversion.2.in
0
ooiwt@pe119:~/as07-weitsang$ cat inputs/inversion.3.in
1
-100
ooiwt@pe119:~/as07-weitsang$ ./inversion < inputs/inversion.3.in
0
```
