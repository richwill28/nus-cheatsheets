# Question 1: Walk (10 marks)
-----------------------------

Many cities in the world, such as New York City, are laid
out as a grid, with streets running in the north-south
directions and east-west directions.  Buildings between two
streets are called _blocks_.

Suppose we start at a junction of two streets, and we wish
to walk to another junction that is $y$ blocks to the north
and $x$ blocks to the east, how many possible paths are
there to walk to the destination?

For example, suppose we want to walk to a junction that is
one block to the north and one block to the east, and we can
only walk eastward and northward, there are two possible
paths.  We can walk east for one block, then walk north for
another block.  Or, we can walk north first for one block,
then walk east for another block.

The figure below illustrates the scenario, where we are
positioned at source D and we wish to go to destination B.
The two possible paths are DAB and DEB.

```
A---B---C
|   |   |
D---E---F
```

Suppose now we want to walk to C, a junction that is two
blocks to the east, and one block to the north.  There are
three different possible paths.  The three paths are DABC,
DEBC, DEFC.

Write a program `walk` that reads in two integers x and y (x
>= 0, y >= 0), and prints the number of possible paths we
can walk to the destination that is x block to the east and
y block to the north.

### Grading Criteria

The grading criteria for this question is:

|              | Marks |
|--------------|-------|
| Correctness  | 5     |
| Efficiency   | 5     |

Your solution must take no more than O(xy) time to obtain
the full efficiency marks.

Hint: Think recursively, but solve iteratively.

## Sample Run
-------------
```
ooiwt@pe111:~/as08-ooiwt$ walk
10 0
1
ooiwt@pe111:~/as08-ooiwt$ walk
1 1
2
ooiwt@pe111:~/as08-ooiwt$ walk
2 2
6
```

# Question 2: Maze (20 marks)
-----------------------------

Agent Scully woke up and found herself in the dark.  She
figured out that she is in a maze.  She has to find her way
out if there is one!

The maze can be simplified as a grid of (m x n) cells. Each
cell can be of the following:

1. '#' denotes a segment of a wall.  No one can pass through
   the walls.

2. '.' denotes an empty space

3. '@' denotes where Scully is at currently. It is also an
   empty space

Anytime Scully reaches a border cell (a cell in either the
top-most or bottom-most row or left-most or right-most
column), she escapes the maze and can go save her partner
Agent Mulder.  She can only move from one empty space to
another adjacent cell in one step.  Two cells are adjacent
if they share a common edge.

Scully took CS1010, and she got a concrete plan to seek a
way out by trial and error.  She follows **strictly** the
following strategy to find a way through the maze starting
from her initial position.  At each time step,

1. She looks for an empty adjacent cell that has never been
visited yet, in the sequence of up/right/down/left to the
current cell she is at.  If there is an empty adjacent cell,
she moves to that cell.  The cell she moves to is now
visited.

2. If no empty, unvisited, adjacent cell exists, she
backtracks on the path that she comes from, moving one step
back, and repeat 1 again.

In this way, Scully will systematically explore the maze,
with no repetitive visit of any cell more than once except
when she backtracks.  She will stop when successfully
escaped the maze, or finds that there is no way out after
backtracking all the way back to the original position.  She
is completely trapped within the maze and now must wait for
Agent Mulder to come and free her.

Write a program `maze.c`, that reads from standard input.
First, read two positive integers m and n, followed by m
lines of n characters in each line that represents the maze
setup.  One and only one `@` will be present in the maze
setup.

The program then prints, to standard output, an animation of
$k$ iterations. The output should only contain $m$ rows with
$n$ characters in each row, with an additional row at last.
Similarly, you must use `#` to represent a wall, a `.` to
represents empty space, and `@` to represent where Scully is
at.  After printing the maze, your program prints the number
of steps that Scully has made.

You should use recursion to explore the maze and look for a
way out.

Here is an example.  The following is the starting position
of Scully and the maze.

```
###########
#.#.....#.#
#.#####.#.#
#.#...#.#.#
#.#.#.....#
#.#.##.#.##
#@#.#..#..#
#...#.#.#..
###########
0
```

Scully firstly moves five steps up:
```
###########
#@#.....#.#
#.#####.#.#
#.#...#.#.#
#.#.#.....#
#.#.##.#.##
#.#.#..#..#
#...#.#.#..
###########
5
```

At this point, Scully is stuck since there is no more
adjacent empty cell that has not been visited.  Scully then
backtracks:
```
###########
#.#.....#.#
#.#####.#.#
#.#...#.#.#
#.#.#.....#
#.#.##.#.##
#@#.#..#..#
#...#.#.#..
###########
10
```

Scully then moves down:
```
###########
#.#.....#.#
#.#####.#.#
#.#...#.#.#
#.#.#.....#
#.#.##.#.##
#.#.#..#..#
#@..#.#.#..
###########
11
```

Then right:
```
###########
#.#.....#.#
#.#####.#.#
#.#...#.#.#
#.#.#.....#
#.#.##.#.##
#.#.#..#..#
#..@#.#.#..
###########
13
```

Then up:
```
###########
#.#.....#.#
#.#####.#.#
#.#@..#.#.#
#.#.#.....#
#.#.##.#.##
#.#.#..#..#
#...#.#.#..
###########
17
```

Then right (two steps) and then down (two steps) and then
right (two steps):

```
###########
#.#.....#.#
#.#####.#.#
#.#...#.#.#
#.#.#..@..#
#.#.##.#.##
#.#.#..#..#
#...#.#.#..
###########
22
```

Then Scully moves up and left, and she is stuck again.

```
###########
#.#@....#.#
#.#####.#.#
#.#...#.#.#
#.#.#.....#
#.#.##.#.##
#.#.#..#..#
#...#.#.#..
###########
29
```

At this point she backtracks:

```
###########
#.#.....#.#
#.#####.#.#
#.#...#.#.#
#.#.#..@..#
#.#.##.#.##
#.#.#..#..#
#...#.#.#..
###########
36
```

Moves right, and up, and stuck again!

```
###########
#.#.....#@#
#.#####.#.#
#.#...#.#.#
#.#.#.....#
#.#.##.#.##
#.#.#..#..#
#...#.#.#..
###########
41
```
She backtracks again,

```
###########
#.#.....#.#
#.#####.#.#
#.#...#.#.#
#.#.#...@.#
#.#.##.#.##
#.#.#..#..#
#...#.#.#..
###########
45
```

This time she found her way out!

```
###########
#.#.....#.#
#.#####.#.#
#.#...#.#.#
#.#.#.....#
#.#.##.#.##
#.#.#..#..#
#...#.#.#.@
###########
50
```

It took her a total of 50 steps to exit the maze.

### Grading Criteria

The grading criteria for this question is:

|              | Marks |
|--------------|-------|
| Correctness  | 10    |
| Efficiency   | 10    |

Your solution must take no more than O(mn) time to obtain
the full efficiency marks.

Hint: You need to strictly follow the described strategy and
sequence of exploration. Do not forget to print the initial
matrix and the final matrix.


### Animation on Screen

We have provided the code to draw the maze in the skeleton
file.

## Sample Run
-------------

You can use the sample program `~cs1010/maze` on the given
inputs to view the animation.
