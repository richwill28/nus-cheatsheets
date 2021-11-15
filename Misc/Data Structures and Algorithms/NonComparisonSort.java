public class NonComparisonSort {
    // Complexities below assume n items to be sorted, with keys of size k, 
    // digit size d, numbers of base b, and r the range of numbers to be sorted.

    /**
     * Get the maximum number in an array.
     * 
     * @param arr An array of integer.
     * @param size The size of the array.
     * @return The maxiumum number in the array.
     */
    private static int getMaxNumber(int[] arr, int size) {
        int currentMax = arr[0];
        for (int i = 0; i < size; i++) {
            if (arr[i] > currentMax) {
                currentMax = arr[i];
            }
        }

        return currentMax;
    }

    /**
     * Modified counting sort for radix which preserves stability. 
     * Time complexity: O(n + b)
     * 
     * @param arr An array of integer.
     * @param n The number of items in the array.
     * @param exp The digit exponent.
     */
    private static void countingSortForRadix(int[] arr, int n, int exp) {
        int[] sortedArr = new int[n];
        int[] countingArr = new int[10];
        for (int i = 0; i < n; i++) {
            int digit = arr[i] / exp % 10;
            countingArr[digit]++;
        }

        for (int i = 1; i < 10; i++) {
            countingArr[i] += countingArr[i - 1];    // update to the actual index of each digit
        }

        // iterate from right to left of the array
        for (int i = n - 1; i >= 0; i--) {
            int digit = arr[i] / exp % 10;
            int sortedIndex = countingArr[digit] - 1;    // the correct index after sorting
            sortedArr[sortedIndex] = arr[i];
            countingArr[digit]--;    // adjust for the correct next index
        }

        for (int i = 0; i < n; i++) {
            arr[i] = sortedArr[i];    // copy back
        }
    }

    /**
     * Random: O(n + r) 
     * Ascending: O(n + r) 
     * Descending: O(n + r)
     * 
     * @param arr An array of integer.
     * @param n The number of items in the array.
     */
    public static void countingSort(int[] arr, int n) {
        int r = getMaxNumber(arr, n);
        int[] countingArr = new int[r + 1];
        for (int i = 0; i < n; i++) {
            countingArr[arr[i]]++;
        }

        int currentIndex = 0;
        for (int i = 0; i < r + 1; i++) {
            while (countingArr[i] != 0) {
                arr[currentIndex++] = i;
                countingArr[i]--;
            }
        }
    }

    /**
     * Random: O(d * (n + b))
     * Ascending: O(d * (n + b))
     * Descending: O(d * (n + b))
     * 
     * @param arr An array of integer.
     * @param n The number of items in the array.
     */
    public static void radixSort(int[] arr, int n) {
        int r = getMaxNumber(arr, n);
        for (int i = 1; r / i > 0; i *= 10) {
            countingSortForRadix(arr, n, i);
        }
    }
}
