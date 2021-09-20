/**
 * CS1010 Semester 1 AY20/21
 * Assignment 9: Digits
 *
 * A program to recognise handwritten digits using k-nearest
 * neighbors algorithm.
 *
 * @file: digits.c
 * @author: Richard Willie (Group C08)
 */
#include "cs1010.h"

typedef struct {
  long label;
  char **image;
} digit;

typedef struct {
  long number;
  long distance;
} neighbor;

/**
 * Release memory for 2D long array.
 *
 * @param[in] any The 2D array.
 * @param[in] len The length of the array.
 */
void release_long(long **any, long len) {
  for (long i = 0; i <= len - 1; i += 1) {
    free(any[i]);
  }
  free(any);
}

/**
 * Release memory for 2D neighbor array.
 *
 * @param[in] any The 2D array.
 * @param[in] len The length of the array.
 */
void release_neighbor(neighbor **any, long len) {
  for (long i = 0; i <= len - 1; i += 1) {
    free(any[i]);
  }
  free(any);
}

/**
 * Print the recognition results.
 *
 * @param[in] test The test digits.
 * @param[in] index The recognition result.
 * @param[in] i The i-th test image.
 */
void print_result(digit *test, long *index, long i) {
  cs1010_print_long(test[i].label);
  cs1010_print_string(" ");
  cs1010_println_long(index[i]);
}

/**
 * Calculate the recognition results and accuracy.
 *
 * @param[in,out] test The test digits.
 * @param[in,out] five The five nearest neighbors.
 * @param[in,out] count The frequency of each neighbor.
 * @param[in] m The number of test digits.
 */
void calculate_result(digit *test, neighbor **five, long **count, long m) {
  // An array to store the recognition results. 
  long *index = malloc((size_t)m * sizeof(long));
  if (index != NULL) {
    double accuracy = 0;

    for (long i = 0; i <= m - 1; i += 1) {
      long most = count[i][0];
      long five_one = 0;
      long two_two = 0;
      long two[2] = {0};

      for (long j = 0; j <= 9; j += 1) {
        if (count[i][j] > most) {
          most = count[i][j];
          index[i] = j;
        }

        // Check if there are same frequencies of digits that could result in a tie.
        if (count[i][j] == 1) {
          five_one += 1;
        } else if (count[i][j] == 2) {
          two[two_two] = j;
          two_two += 1;
        }
      }

      // Tie breaker.
      if (five_one == 5) {
        index[i] = five[i][0].number;
      } else if (two_two == 2) {
        long flag = 0;

        for (long j = 0; j <= 4; j += 1) {
          if (flag == 0 && (five[i][j].number == two[0] || five[i][j].number == two[1])) {
            index[i] = five[i][j].number;
            flag += 1;
          }
        }
      }

      // If recognised succesfully, increase accuracy.
      if (test[i].label == index[i]) {
        accuracy += 1;
      }

      // Print the results.
      print_result(test, index, i);
    }

    // Print the accuracy.
    cs1010_println_double((accuracy / m) * 100);
  }

  // Release memory.
  free(test);
  release_neighbor(five, m);
  release_long(count, m);
  free(index);
}

/**
 * Find the five nearest neighbors.
 *
 * @param[in,out] test The test digits.
 * @param[in,out] sorted_data The sorted neighbor based on distance.
 * @param[in] m The number of test digits.
 */
void five_neighbor(digit *test, neighbor **sorted_data, long m) {
  // Initialize an array to store the five nearest neighbors.
  neighbor **five = malloc((size_t)m * sizeof(neighbor *));
  if (five == NULL) {
    free(five);
    return;
  }

  for (long i = 0; i <= m - 1; i += 1) {
    five[i] = malloc((size_t)5 * sizeof(neighbor));
    if (five[i] == NULL) {
      return;
    }
  }

  // Store the five nearest neighbors.
  for (long i = 0; i <= m - 1; i += 1) {
    for (long j = 0; j <= 4; j += 1) {
      five[i][j].number = sorted_data[i][j].number;
      five[i][j].distance = sorted_data[i][j].distance;
    }
  }

  // Initialize an array to store the frequency of each digits from the five neighbors.
  long **count = malloc((size_t)m * sizeof(long *));
  if (count == NULL) {
    free(count);
    return;
  }

  for (long i = 0; i <= m - 1; i += 1) {
    count[i] = calloc((size_t)10, sizeof(long));
    if (count[i] == NULL) {
      return;
    }
  }

  // Count the frequencies of each digits.
  for (long i = 0; i <= m - 1; i += 1) {
    for (long j = 0; j <= 4; j += 1) {
      count[i][five[i][j].number] += 1;
    }
  }

  // Calculate the recognition results.
  calculate_result(test, five, count, m);

  // Release memory.
  release_neighbor(sorted_data, m);
}

/**
 * Find maximum in an array.
 *
 * @param[in,out] data The list of neighbors.
 * @param[in] m The number of test digits.
 * @param[in] n The number of training digits.
 *
 * @return Return an array of max values for each test digit.
 */
long *find_max(neighbor **data, long m, long n) {
  long *max = malloc((size_t)m * sizeof(long));
  if (max == NULL) {
    free(max);
    return NULL;
  }

  for (long i = 0; i <= m - 1; i += 1) {
    max[i] = data[i][0].distance;
    for (long j = 0; j <= n - 1; j += 1) {
      if (data[i][j].distance > max[i]) {
        max[i] = data[i][j].distance;
      }
    }
  }

  return max;
}

/**
 * The sorting algorithm.
 * 
 * @param[in,out] count The frequency array.
 * @param[in] max The max distances array.
 * @param[in] data The list of neighbors.
 * @param[in,out] sorted_data The array to store the sorted neighbors.
 * @param[in] m The number of test digits.
 * @param[in] n The number of training digits.
 */
void run_sort(long **count, long *max, neighbor **data, neighbor **sorted_data, long m, long n) {
  // Count the frequency of each distance.
  for (long i = 0; i <= m - 1; i += 1) {
    for (long j = 0; j <= n - 1; j += 1) {
      count[i][data[i][j].distance] += 1;
    }
  }

  // Initialize indexes.
  for (long i = 0; i <= m - 1; i += 1) {
    for (long j = 1; j <= max[i]; j += 1) {
      count[i][j] += count[i][j - 1];
    }
  }

  // Store the sorted neighbors.
  for (long i = 0; i <= m - 1; i += 1) {
    for (long j = n - 1; j >= 0; j -= 1) {
      sorted_data[i][count[i][data[i][j].distance] - 1].distance = data[i][j].distance;
      sorted_data[i][count[i][data[i][j].distance] - 1].number = data[i][j].number;
      count[i][data[i][j].distance] -= 1;
    }
  }
}

/**
 * Sort all the neighbors based on distance.
 *
 * @param[in,out] test The test digits.
 * @param[in,out] data The list of neighbors.
 * @param[in] m The number of test digits.
 * @param[in] n The number of training digits.
 */
void sort_distance(digit *test, neighbor **data, long m, long n) {
  // Find the max distance from each test image.
  long *max = find_max(data, m, n);
  if (max == NULL) {
    free(max);
    return;
  }
  // Initialize an array to count the frequency of each distance.
  long **count = malloc((size_t)m * sizeof(long *));
  if (count == NULL) {
    free(count);
    return;
  }

  for (long i = 0; i <= m - 1; i += 1) {
    count[i] = calloc((size_t)(max[i] + 1), sizeof(long));
    if (count[i] == NULL) {
      return;
    }
  }

  // Initialize an array to store the sorted neighbors.
  neighbor **sorted_data = malloc((size_t)m * sizeof(neighbor *));
  if (sorted_data == NULL) {
    free(sorted_data);
    return;
  }

  for (long i = 0; i <= m - 1; i += 1) {
    sorted_data[i] = malloc((size_t)n * sizeof(neighbor));
    if (sorted_data[i] == NULL) {
      return;
    }
  }

  // Run the sorting algorithm.
  run_sort(count, max, data, sorted_data, m, n);

  // Find five nearest neighbors.
  five_neighbor(test, sorted_data, m);

  // Release memory.
  release_neighbor(data, m);
  free(max);
  release_long(count, m);
}

/**
 * Calculate the distances of all training images.
 *
 * @param[in,out] test The test digits.
 * @param[in,out] train The training digits.
 * @param[in] m The number of test digits.
 * @param[in] n The number of training digits.
 */
void calculate_distance(digit *test, digit *train, long m, long n) {
  // Initialize neighbor datas and check for memory allocation error.
  neighbor **data = malloc((size_t)m * sizeof(neighbor *));
  if (data == NULL) {
    free(data);
    return;
  }

  for (long i = 0; i <= m - 1; i += 1) {
    data[i] = malloc((size_t)n * sizeof(neighbor));
    if (data[i] == NULL) {
      return;
    }
  }

  // Calculate the distances of each training image.
  for (long i = 0; i <= m - 1; i += 1) {
    for (long j = 0; j <= n - 1; j += 1) {
      long count = 0;

      for (long k = 0; k <= 27; k += 1) {
        for (long l = 0; l <= 27; l += 1) {
          if (test[i].image[k][l] != train[j].image[k][l]) {
            count += 1;
          }
        }
      }

      data[i][j].number = train[j].label;
      data[i][j].distance = count;
    }
  }

  // Sort the neighbor datas based on distance.
  sort_distance(test, data, m, n);

  // Release memory.
  free(train);
}

int main() {
  long n = cs1010_read_long();

  // Initialize training images and check for memory allocation error.
  digit *train = malloc((size_t)n * sizeof(digit));
  if (train == NULL) {
    free(train);
    return -1;
  }

  for (long i = 0; i <= n - 1; i += 1) {
    train[i].label = cs1010_read_long();
    train[i].image = cs1010_read_word_array(28);
  }

  long m = cs1010_read_long();

  // Initialize test images and check for memory allocation error.
  digit *test = malloc((size_t)m * sizeof(digit));
  if (test == NULL) {
    free(test);
    return -1;
  }

  for (long i = 0; i <= m - 1; i += 1) {
    test[i].label = cs1010_read_long();
    test[i].image = cs1010_read_word_array(28);
  }

  // Calculate the distances of each training image.
  calculate_distance(test, train, m, n);
  return 0;
}
