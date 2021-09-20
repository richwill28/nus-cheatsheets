/**
 * CS1010 Semester 1 AY20/21
 * Assignment 7: Peak
 *
 * A program to find the peak of a given array of numbers. We assumed
 * the array to be first increasing and then decreasing. It is also
 * possible that the array is non-increasing, non-decreasing, or there
 * is a flat plateau where there are multiple peaks with uniform elevation.
 * In these cases, a peak does not exist.
 *
 * @file: peak.c
 * @author: Richard Willie (Group C08)
 */
#include "cs1010.h"
#include <stdbool.h>

/**
 * Find the position of a peak from a given array.
 *
 * @param[in] first The first positon of an array.
 * @param[in] last The last position of an array.
 * @param[in] arr An array of numbers.
 *
 * @return Return the position of the peak of a given array.
 */
long peak(long first, long last, long *arr) {
  /* Base Cases */

  // Only one element in the array.
  if (last == first) {
    return first;
  }

  // Only two elements in the array.
  if (last == first + 1) {
    if (arr[first] >= arr[last]) {
      return first;
    }
    return last;
  }

  // The position of the middle element in the array.
  long mid = first + (last - first) / 2;

  // If both neighbors is lower than current element, then it is a peak.
  if (arr[mid] > arr[mid + 1] && arr[mid] > arr[mid - 1]) {
    return mid;
  }

  /* Recursive Cases */

  // A variable to store the postition of the peak of a given array.
  long pos;

  if (arr[mid] > arr[mid + 1] && arr[mid] < arr[mid - 1]) {
    // Left side is higher, then the peak must be located on the left.
    pos = peak(first, mid - 1, arr);
  } else if (arr[mid] > arr[mid - 1] && arr[mid] < arr[mid + 1]) {
    // Right side is higher, then the peak must be located on the right.
    pos = peak(mid + 1, last, arr);
  } else {
    /*
     Handle cases where there are equal elevations in the array. The method is
     by separating the array into two sides recursively until non-distinct
     elements are no longer present or until base cases are reach.
     */

    // Initialize variables to store the peak of the left and right arrays.
    long left_peak = peak(first, mid, arr);
    long right_peak = peak(mid + 1, last, arr);

    if (arr[left_peak] >= arr[right_peak]) {
      pos = left_peak;
    } else {
      pos = right_peak;
    }
  }

  return pos;
}

/**
 * Return true if an array has a peak, otherwise return false.
 *
 * @param[in] n The length of the array.
 * @param[in] peak_pos The position of the peak of an array.
 * @param[in] arr An array of numbers.
 *
 * @return Return true or false.
 */
bool is_peak(long n, long peak_pos, long *arr) {
  bool value = true;
  
  if (arr[peak_pos] == arr[0] || arr[peak_pos] == arr[n - 1]) {
    // The array is either non-increasing or non-decreasing.
    value = false;
  } else if (arr[peak_pos] == arr[peak_pos - 1] || arr[peak_pos] == arr[peak_pos + 1]) {
    // The so-called peak is a flat plateau.
    value = false;
  }

  return value;
}

int main() {
  // Read the length of an array.
  long n = cs1010_read_long();

  // Read an array and check for memory allocation error.
  long *arr = cs1010_read_long_array(n);
  if (arr == NULL) {
    return -1;
  }

  // A variable to store the position of the peak of a given array.
  long peak_pos = peak(0, n - 1, arr);

  // If a peak exists print its position, otherwise print 'no peak'.
  if (is_peak(n, peak_pos, arr)) {
    cs1010_println_long(peak_pos);
  } else {
    cs1010_println_string("no peak");
  }

  // Release the memory allocated for the array.
  free(arr);
  return 0;
}
