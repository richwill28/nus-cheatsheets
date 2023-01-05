## Part 4: Accurate Performance Analysis (Lab Submission)

### Exercise 7

Change this line..

```c
        result.element[i][j] += a.element[i][k] * b.element[k][j];
```

to this line..

```c
        result.element[i][k] += a.element[i][j] * b.element[j][k];
```

### Exercise 8

Two key observations:

- Row-major access is faster than column-major access.
- Row-major access has less L1D cache misses as compared to column-major access.

The results above should not be a surprise. Caches are organized into cache lines, and they are usually 64 bytes in most modern processors. The size of cache line is important with respect to spatial cache locality (e.g., how close the data are stored in memory). For instance, with row-major implementation, access is usually fast because elements are in the same cache line.

For column major access, on the other hand, consecutive elements might not be in the same cache line (due to spatial locality). Even when the CPU is able to prefetch more than one cache line at a time, access could still be sloww. For example, for large matrix, we might end up evicting entries from the cache. So that by the time we get back to the first row, the entries have already been evicted from cache, and we have to load them again from memory.
