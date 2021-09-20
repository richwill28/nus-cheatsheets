/**
 * The Optimization class contains a static routine to find the maximum in an array that changes direction at most once.
 */
public class Optimization {
    /**
     * A set of test cases.
     */
    static int[][] testCases = {
            {1, 3, 5, 7, 9, 11, 10, 8, 6, 4},
            {67, 65, 43, 42, 23, 17, 9, 100},
            {4, -100, -80, 15, 20, 25, 30},
            {2, 3, 4, 5, 6, 7, 8, 100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83}
    };

    /**
     * Returns the maximum item in the specified array of integers which changes direction at most once.
     *
     * @param dataArray an array of integers which changes direction at most once.
     * @return the maximum item in data Array
     */
    public static int searchMax(int[] dataArray) {
        // Assumes no duplicate elements
        int maxElement = Integer.MIN_VALUE;
        try {
            if (dataArray.length == 0) {
                throw new IllegalArgumentException("Error: EmptyArrayException");
            } else if (dataArray.length < 4) {
                for (int data : dataArray) {
                    maxElement = Math.max(data, maxElement);
                }
                return maxElement;
            }
            boolean isAscendingLeft = dataArray[0] < dataArray[1];
            boolean isAscendingRight = dataArray[dataArray.length - 2] < dataArray[dataArray.length - 1];
            if (isAscendingLeft && isAscendingRight) {
                // array has ascending-ascending pattern
                return dataArray[dataArray.length - 1];
            } else if (isAscendingLeft && !isAscendingRight) {
                // array has ascending-descending pattern
                return binarySearchMax(dataArray);
            } else if (!isAscendingLeft && isAscendingRight) {
                // array has descending-ascending pattern
                return Math.max(dataArray[0], dataArray[dataArray.length - 1]);
            } else if (!isAscendingLeft && !isAscendingRight) {
                // array has descending-descending pattern
                return dataArray[0];
            } else {
                // if the evaluations above somehow failed
                throw new RuntimeException("Error: EvaluationFailedException");
            }
        } catch (IllegalArgumentException e) {
            // Do nothing
            System.out.println(e.getMessage());
        } catch (RuntimeException e) {
            // Do nothing
            System.out.println(e.getMessage());
        }
        return maxElement;
    }

    public static int binarySearchMax(int[] dataArray) {
        int maxElement = Integer.MIN_VALUE;
        int leftIndex = 0;
        int rightIndex = dataArray.length - 1;
        while (leftIndex < rightIndex) {
            int middleIndex = leftIndex + (rightIndex - leftIndex) / 2;
            if (dataArray[middleIndex - 1] > dataArray[middleIndex]) {
                // recurse to the left subarray
                rightIndex = middleIndex;
            } else if (dataArray[middleIndex - 1] < dataArray[middleIndex]) {
                // recurse to the right subarray
                leftIndex = middleIndex + 1;
            }
            maxElement = Math.max(dataArray[middleIndex], maxElement);
        }
        return maxElement;
    }

    /**
     * A routine to test the searchMax routine.
     */
    public static void main(String[] args) {
        for (int[] testCase : testCases) {
            System.out.println(searchMax(testCase));
        }
    }
}
