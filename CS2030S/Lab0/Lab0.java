import java.util.Scanner;

/**
 * CS2030S Lab 0: Estimating Pi with Monte Carlo
 * Semester 2, 2020/21
 * <p>
 * This program takes in two command line arguments: the
 * number of points and a random seed.  It runs the
 * Monte Carlo simulation with the given argument and print
 * out the estimated pi value.
 *
 * @author richwill28
 */

class Lab0 {
    private static double estimatePi(long numOfPoints, int seed) {
        double n = 0;
        double k = numOfPoints;
        Circle circle = new Circle(new Point(0.5, 0.5), 0.5);
        RandomPoint.setSeed(seed);
        for (int i = 0; i < numOfPoints; i++) {
            Point point = new RandomPoint(0, 1, 0, 1);
            n += circle.contains(point) ? 1 : 0;
        }
        return 4 * n / k;
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int numOfPoints = sc.nextInt();
        int seed = sc.nextInt();

        double pi = estimatePi(numOfPoints, seed);

        System.out.println(pi);
        sc.close();
    }
}
