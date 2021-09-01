import java.util.Random;

class RandomPoint extends Point {
    private static Random rng = new Random(1);

    static void setSeed(long seed) {
        rng.setSeed(seed);
    }

    public RandomPoint(double minX, double maxX, double minY, double maxY) {
        super(minX + (maxX - minX) * rng.nextDouble(), minY + (maxY - minY) * rng.nextDouble());
    }
}
