#include <random>

class ComparisonSort {
    // The lower bound for comparison-based sorting algorithm is O(n log n).

    private:
        /**
         * The merging routine of Merge Sort. 
         * Time complexity: O(n)
         * 
         * @param arr An array of integer.
         * @param lowerIndex The lower index of the left subarray.
         * @param middleIndex The upper and lower index of the left and right subarray, respectively.
         * @param upperIndex The upper index of the right subarray.
         */
        void merge(int arr[], int lowerIndex, int middleIndex, int upperIndex) {
            int size = upperIndex - lowerIndex + 1;
            int sortedArr[size];
            int currentIndex = 0;
            int leftIndex = lowerIndex;
            int rightIndex = middleIndex + 1;

            while (leftIndex <= middleIndex && rightIndex <= upperIndex) {
                sortedArr[currentIndex++] = (arr[leftIndex] <= arr[rightIndex]) ? arr[leftIndex++] : arr[rightIndex++];    // the merging process
            }

            while (leftIndex <= middleIndex) {
                sortedArr[currentIndex++] = arr[leftIndex++];    // leftover, if any
            }

            while (rightIndex <= upperIndex) {
                sortedArr[currentIndex++] = arr[rightIndex++];    // leftover, if any
            }

            for (int i = 0; i < size; i++) {
                arr[lowerIndex + i] = sortedArr[i];    // copy back
            }
        }

        /**
         * The partitioning routine of Quick Sort. 
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
                if (arr[i] < pivot) {
                    currentIndex++;
                    std::swap(arr[i], arr[currentIndex]);
                }
            }
            std::swap(arr[lowerIndex], arr[currentIndex]);
            return currentIndex;
        }

        /**
         * Generate uniformly distributed integer on the closed 
         * interval [lowerBound, upperBound].
         * 
         * @param lowerBound Lower bound of the interval.
         * @param upperBound Upper bound of the interval.
         * @return The random number generated.
         */
        int randomNumberGenerator(int lowerBound, int upperBound) {
            std::random_device rd;    // obtain a seed for the random number engine
            std::mt19937 gen(rd());    // mersenne twister engine seeded with rd()
            std::uniform_int_distribution<int> distrib(lowerBound, upperBound);
            return distrib(gen);    // transform the random unsigned int generated by gen into an int
        }

        /**
         * The partitioning routine of Randomized Quick Sort. 
         * Time complexity: O(n)
         * 
         * @param arr An array of integer.
         * @param lowerIndex The lower index of the subarray.
         * @param upperIndex The upper index of the subarray.
         * @return The index of the pivot.
         */
        int randomizedPartition(int arr[], int lowerIndex, int upperIndex) {
            int randomIndex = randomNumberGenerator(lowerIndex, upperIndex);
            std::swap(arr[lowerIndex], arr[randomIndex]);
            int pivot = arr[lowerIndex];
            int currentIndex = lowerIndex;
            for (int i = lowerIndex + 1; i <= upperIndex; i++) {
                if (arr[i] < pivot) {
                    currentIndex++;
                    std::swap(arr[i], arr[currentIndex]);
                }
            }
            std::swap(arr[lowerIndex], arr[currentIndex]);
            return currentIndex;
        }
    
    public:
        /**
         * Random: O(n^2) 
         * Ascending: O(n) 
         * Descending: O(n^2)
         * 
         * @param arr An array of integer.
         * @param size The size of the array.
         */
        void bubbleSort(int arr[], int size) {
            bool isSwapped;
            int indexOfLastUnsortedElement = size - 1;
            do {
                isSwapped = false;
                for (int i = 0; i < indexOfLastUnsortedElement; i++) {
                    if (arr[i] > arr[i + 1]) {
                        std::swap(arr[i], arr[i + 1]);
                        isSwapped = true;
                    }
                }
                indexOfLastUnsortedElement--;
            } while (isSwapped);
        }

        /**
         * Random: O(n^2) 
         * Ascending: O(n) 
         * Descending: O(n^2)
         * 
         * @param arr An array of integer.
         * @param size The size of the array.
         */
        void insertionSort(int arr[], int size) {
            int indexOfElementToBeInserted;
            for (int i = 1; i < size; i++) {
                indexOfElementToBeInserted = i;
                for (int j = i - 1; j >= 0; j--) {
                    if (arr[j] > arr[indexOfElementToBeInserted]) {
                        std::swap(arr[j], arr[indexOfElementToBeInserted]);
                        indexOfElementToBeInserted = j;
                    } else {
                        break;
                    }
                }
            }
        }

        /**
         * Random: O(n log n) 
         * Ascending: O(n log n) 
         * Descending: O(n log n)
         * 
         * @param arr An array of integer.
         * @param lowerIndex The lower index of the subarray.
         * @param upperIndex The upper index of the subarray.
         */
        void mergeSort(int arr[], int lowerIndex, int upperIndex) {
            if (lowerIndex < upperIndex) {
                int middleIndex = lowerIndex + (upperIndex - lowerIndex) / 2;    // an overflow-safe implementation of (lowerIndex + upperIndex) / 2
                mergeSort(arr, lowerIndex, middleIndex);
                mergeSort(arr, middleIndex + 1, upperIndex);
                merge(arr, lowerIndex, middleIndex, upperIndex);
            }
        }

        /**
         * Random: O(n log n) 
         * Ascending: O(n^2) 
         * Descending: O(n^2)
         * 
         * @param arr An array of integer.
         * @param lowerIndex The lower bound of the subarray.
         * @param upperIndex The upper bound of the subarray.
         */
        void quickSort(int arr[], int lowerIndex, int upperIndex) {
            if (lowerIndex < upperIndex) {
                int pivotIndex = partition(arr, lowerIndex, upperIndex);
                quickSort(arr, lowerIndex, pivotIndex - 1);    // recursively sort left subarray
                quickSort(arr, pivotIndex + 1, upperIndex);    // recursively sort right subarray
            }
        }

        /**
         * Random: O(n log n) 
         * Ascending: O(n log n) 
         * Descending: O(n log n)
         * 
         * @param arr An array of integer.
         * @param lowerIndex The lower bound of the subarray.
         * @param upperIndex The upper bound of the subarray.
         */
        void randomizedQuickSort(int arr[], int lowerIndex, int upperIndex) {
            if (lowerIndex < upperIndex) {
                int pivotIndex = randomizedPartition(arr, lowerIndex, upperIndex);
                randomizedQuickSort(arr, lowerIndex, pivotIndex - 1);    // recursively sort left subarray
                randomizedQuickSort(arr, pivotIndex + 1, upperIndex);    // recursively sort right subarray
            }
        }

        /**
         * Random: O(n^2) 
         * Ascending: O(n^2) 
         * Descending: O(n^2)
         * 
         * @param arr An array of integer.
         * @param size The size of the array.
         */
        void selectionSort(int arr[], int size) {
            for (int i = 0; i < size - 1; i++) {
                int indexOfCurrentMinimum = i;
                for (int j = i + 1; j < size; j++) {
                    if (arr[j] < arr[indexOfCurrentMinimum]) {
                        indexOfCurrentMinimum = j;
                    }
                }
                std::swap(arr[i], arr[indexOfCurrentMinimum]);
            }
        }
};
