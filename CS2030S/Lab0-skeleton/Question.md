# Lab 0: Circle and Points

- Deadline: 27 January 2021, Wednesday, 23:59 SST
- Marks: 0

## Prerequisite:

- Familiar with the CS2030S lab guidelines
- Able to access the CS2030S programming environment via ssh
- Completed basic `vim` lessons

## Estimating Pi using Monte Carlo Method

The Monte Carlo method for estimating the value of pi is as
follows.  We have a square of width 2r, and within it, a
circle with a radius of r.

We randomly generate k points within the square.  We count
how many points fall within the circle.  Suppose n points
out of k fall within the circle.

Since the area of the square is 4r^2 and the area of the
circle is pi r^2, the ratio between them is pi/4.  The ratio
n/k should therefore be pi/4, and pi can be estimated as
4n/k.

### Background: Random Number Generator

To estimate pi using the method above, we need to use a
random number generation.  A random number generator is an
entity that spews up one random number after another.  We,
however, cannot generate a truly random number
algorithmically.  We can only generate a pseudo-random
number.  A pseudo-random number generator can be initialized
with a seed.  A pseudo-random number generator, when
initialized with the same seed, always produces the same
sequence of (seemingly random) numbers.

Java provides a class `java.util.Random` that encapsulates a
pseudo-random number generator. We can create a random
number generator with a seed:

```
Random rng = new Random(1);
```

We can then call `rng.nextDouble()` repeatedly to generate
random numbers between 0 and 1.

Using a fixed seed is important for testing since the
execution of the program will be deterministic, even when
random numbers are involved.

## Files Provided

In this directory, you should see the following files:

- Skeleton Java files: `Point.java`, `RandomPoint.java`,
  `Circle.java`, `Lab0.java`

- Inputs and outputs for `Lab0`: `inputs/Lab0.k.in` and
  `outputs/Lab0.k.out` for different values of k.

- Bash script: `test.sh` for testing `Lab0` if it estimates
  pi correctly, by comparing the output when running `Lab0` on
  `inputs/Lab0.k.in` to the expected output in
  `outputs/Lab0.k.out`

- Unit tests for Java classes: `Test.java`.  This file tests
  individual classes to check if they have the expected
  behavior

## Your Task

A skeleton code has been given.  Your task is to complete
the implementation of the classes `Point`, `RandomPoint`,
`Circle`, and `Lab0`, according to the OO principles that
were taught: abstraction, encapsulation, information hiding,
inheritance, tell-don't-ask.

Another client class `Test` is given to test the behavior of
your code.  You are required to pass the test cases given in
`Test`.

## `Circle`

Most of the `Circle` class has been written for you.  You
need to complete the method `contains`.

## `Point`

You need to fill in the class `Point` with the constructor,
`toString`, and any other methods necessary.

## `RandomPoint`

`RandomPoint` is a subclass of `Point` that represents a
randomly generated point.  The random number generator that
generates a random point has a default seed of 1.  There is
a public method `setSeed()` that we can use to update the
seed.

This is how it can be used.

To generate a new point,
```
Point p = new RandomPoint(minX, maxX, minY, maxY); 
```

`minX`, `minY`, `maxX`, `maxY` represent the minimum and
maximum possible x and y values respectively, for each
randomly generated point.

To set the random seed,
```
RandomPoint.setSeed(10);
```

Tips: What are the fields and methods that should be
associated with the class `RandomPoint` instead of an
instance of `RandomPoint`?

### Lab0

`Lab0` is the main program to solve the problem above.  The
`main` method is provided.  It includes the method to read
in the number of points and the seed from the standard input
and to print the estimated pi value.

The method `estimatePi` is incomplete.  Determine how you
should declare `estimatePi`, then complete the body by
generating random points and count how many fall under the
given circle.

Use r = 0.5 and use `long` and `double` within `estimatePi`
to ensure you have the right precision.

To run `Lab0` and enter the input manually, run
```
java Lab0
```

The program will pause, waiting for inputs from keyboards.
Enter two numbers. The first is the number of points. The
second is the seed.

You can enter the two numbers into a text file, say, `TEST`,
and then run
```
java Lab0 < TEST
```

Sample inputs and outputs have been provided and can be
found under the `inputs` and `outputs` directory.

## Testing

You should test your classes individually before testing
`Lab0`.  A simple `Test.java` is given.  You can run it with
```
java Test
```

To test your implementation of `Lab0`, automatically against
the test data given in `inputs` and `outputs`,
```
./test.sh Lab0
```
