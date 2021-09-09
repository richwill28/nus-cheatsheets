#include <iostream>

class SelectionAlgo {
    private:
        /**
         * The partitioning routine of Quick Select. 
         * Time complexity: O(n)
         * 
         * @param arr An array of integer.
         * @param lowerIndex The lower index of the subarray.
         * @param upperIndex The upper index of the subarray.
         * @return The index of the pivot.
         */
        int partition(int arr[], int lowerIndex, int upperIndex) {
            int pivot = arr[lowerIndex];
            int currentIndex = lowerIndex;
            for (int i = lowerIndex + 1; i <= upperIndex; i++) {
                if (arr[i] <= pivot) {
                    currentIndex++;
                    std::swap(arr[i], arr[currentIndex]);
                }
            }
            std::swap(arr[lowerIndex], arr[currentIndex]);
            return currentIndex;
        }

    public:
        /**
         * Best time complexity: O(n) 
         * Worst time complexity: O(n^2)
         * 
         * @param arr An array of integer.
         * @param lowerIndex The lower index of the subarray.
         * @param upperIndex The upper index of the subarray.
         * @param k Find the kth smallest element.
         * @return The kth smallest element.
         */
        int quickSelect(int arr[], int lowerIndex, int upperIndex, int k) {
            if (k > 0 && k <= upperIndex - lowerIndex + 1) {
                int pivotIndex = partition(arr, lowerIndex, upperIndex);
                int position = pivotIndex - lowerIndex;
                if (position == k - 1) {
                    return arr[pivotIndex];
                } else if (position > k - 1) {
                    // position is more, recur to left side
                    return quickSelect(arr, lowerIndex, pivotIndex - 1, k);
                } else if (position < k - 1) {
                    // position is less, recur to right side
                    return quickSelect(arr, pivotIndex + 1, upperIndex, k - (position + 1));
                }
            }
            return -1;
        }
};
