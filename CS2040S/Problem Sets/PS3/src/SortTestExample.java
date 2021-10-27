/**
 * This class is a simple example for how to use the sorting classes.
 * It sorts three numbers, and measures how long it takes.
 */
public class SortTestExample {
    public static void main(String[] args) {
        // Create three key value pairs
        KeyValuePair[] testArray = new KeyValuePair[3];
        testArray[0] = new KeyValuePair(10, 20);
        testArray[1] = new KeyValuePair(5, 20);
        testArray[2] = new KeyValuePair(8, 20);

        // Create a stopwatch
        StopWatch watch = new StopWatch();
        ISort sortingObject = new SorterA();

        // Do the sorting
        watch.start();
        sortingObject.sort(testArray);
        watch.stop();

        System.out.println(testArray[0]);
        System.out.println(testArray[1]);
        System.out.println(testArray[2]);
        System.out.println("Time: " + watch.getTime());
    }
}
