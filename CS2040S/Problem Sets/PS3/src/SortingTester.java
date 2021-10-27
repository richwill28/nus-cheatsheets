import java.util.Random;

public class SortingTester {
    public static boolean checkSort(ISort sorter, int size) {
        // Using Random class to generate random int for KeyValuePair
        // so that we can take in any size for our testArray
        Random r = new Random();
        r.setSeed(1);
        KeyValuePair[] testArray1 = new KeyValuePair[size];
        for (int i = 0; i < size; i++) {
            testArray1[i] = new KeyValuePair(r.nextInt(), r.nextInt());
        }

        // Using KeyValuePair method compareTo to check if the array
        // is sorted in correct order by Key
        boolean result = true;
        sorter.sort(testArray1);
        for (int i = 0; i < size - 1; i++) {
            if (testArray1[i].compareTo(testArray1[i + 1]) == 1) {
                result = false;
                break;
            }
        }
        return result;
    }

    public static boolean isStable(ISort sorter, int size) {
        // Unable to use a random testArray because testArray
        // must contain duplicate values
        // Create an array that increases then decreases by Key.
        // In the unsorted array, the Value strictly increases
        KeyValuePair[] testArray01 = new KeyValuePair[size];
        for (int i = 0; i < size / 2; i++) {
            testArray01[i] = new KeyValuePair(i, i);
            testArray01[size / 2 + i] = new KeyValuePair(size / 2 - i, size / 2 + i);
        }

        sorter.sort(testArray01);
        boolean result = true;

        for (int i = 0; i < size - 1; i++) {
            // if the Key of the next Pair is the same
            // the Value must increase in order for the
            // sorting algorithm to be stable
            if (testArray01[i].compareTo(testArray01[i + 1]) == 0
                    && testArray01[i].getValue() > testArray01[i + 1].getValue()) {
                result = false;
                break;
            }
        }
        return result;
    }

    public static KeyValuePair[] randomArray(int size) {
        Random r = new Random();
        r.setSeed(1);
        KeyValuePair[] Array = new KeyValuePair[size];
        for (int i = 0; i < size; i++) {
            Array[i] = new KeyValuePair(r.nextInt(), r.nextInt());
        }
        return Array;
    }

    // sortedArray to test best case time complexity
    // SelectionSort time complexity should be similar to average
    public static KeyValuePair[] sortedArray(int size) {
        KeyValuePair[] Array = new KeyValuePair[size];
        for (int i = 0; i < size; i++) {
            Array[i] = new KeyValuePair(i, i);
        }
        return Array;
    }

    // specialArray of [1,3,3,...,3,3,2]
    // used to differentiate BubbleSort and InsertionSort
    // InsertionSort would run at a faster runtime than O(n^2)
    // while SelectionSort will run at an O(n^2) runtime
    public static KeyValuePair[] specialArray(int size) {
        KeyValuePair[] Array = new KeyValuePair[size];
        for (int i = 1; i < size - 1; i++) {
            Array[i] = new KeyValuePair(3, 3);
        }
        Array[0] = new KeyValuePair(1, 1);
        Array[size - 1] = new KeyValuePair(2, 2);
        return Array;
    }

    public static void main(String[] args) {
        // Create a stopwatch
        StopWatch watch1 = new StopWatch();
        StopWatch watch2 = new StopWatch();
        StopWatch watch3 = new StopWatch();
        StopWatch watch4 = new StopWatch();
        StopWatch watch5 = new StopWatch();
        StopWatch watch6 = new StopWatch();
        ISort sortingObject = new SorterA();

        // n = 100
        watch1.start();
        sortingObject.sort(randomArray(100));
        watch1.stop();

        // n = 1000
        watch2.start();
        sortingObject.sort(randomArray(1000));
        watch2.stop();

        // n = 10000
        watch3.start();;
        sortingObject.sort(randomArray(10000));
        watch3.stop();

        // n = 100000
        watch4.start();
        sortingObject.sort(randomArray(100000));
        watch4.stop();

        // n = 10000 sorted
        watch5.start();
        sortingObject.sort(sortedArray(10000));
        watch5.stop();

        // n = 10000 special
        // to differentiate BubbleSort and InsertionSort
        // InsertionSort would run at a faster runtime than O(n^2)
        // while SelectionSort would run at an O(n^2) runtime
        watch6.start();
        sortingObject.sort(specialArray(10000));
        watch6.stop();

        System.out.println("n = 100: " + watch1.getTime());
        System.out.println("n = 1000: " + watch2.getTime());
        System.out.println("n = 10000: " + watch3.getTime());
        System.out.println("n = 100000: " + watch4.getTime());
        System.out.println("n = 10000, sorted: " + watch5.getTime());
        System.out.println("n = 10000, special: " + watch6.getTime());

        // run checkSort 100 times with array size 10000
        // if false is returned, algo is Dr. Evil
        boolean sortable = true;
        for (int i = 0; i < 100; i++) {
            if (! checkSort(sortingObject, 10000)) {
                sortable = false;
                break;
            }
        }
        System.out.println("checkSort: " + sortable);

        // run isStable 100 times with array size 1000
        // if false is returned, algo can be QuickSort or SelectionSort
        boolean stable = true;
        for (int i = 0; i < 100; i++) {
            if (! isStable(sortingObject, 1000)) {
                stable = false;
                break;
            }
        }
        System.out.println("isStable: " + stable);
    }
}
