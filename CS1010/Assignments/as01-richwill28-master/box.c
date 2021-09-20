/**
 * CS1010 Semester 1 AY20/21
 * Assignment 1: Box
 *
 * Read in three positive integer corresponding to the width,
 * height, and length of a box from the standard input, and
 * print the total surface area and the length of the diagonal
 * to the standard output.
 *
 * @file: box.c
 * @author: Richard Willie (Group C08)
 */

#include "cs1010.h"
#include <math.h>

long area_of_rectangle(long first_length, long second_length)
{
  return first_length * second_length;
}

long surface_area(long length, long width, long height)
{
  return 2 * (area_of_rectangle(length, width) + area_of_rectangle(width, height) + area_of_rectangle(height, length));
}

long square(long num)
{
  return num * num;
}

double hypotenuse_of(long length, long width, long height)
{
  return sqrt(square(length) + square(width) + square(height));
}

int main()
{
  long length = cs1010_read_long();
  long width = cs1010_read_long();
  long height = cs1010_read_long();

  cs1010_print_long(surface_area(length, width, height));
  cs1010_print_string(" ");
  cs1010_println_double(hypotenuse_of(length, width, height));

  return 0;
}
