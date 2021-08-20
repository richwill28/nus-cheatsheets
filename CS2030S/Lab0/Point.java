import java.lang.Math;

/**
 * CS2030S Lab 0: Point.java
 * Semester 2, 2020/21
 * <p>
 * The Point class encapsulates a point on a 2D plane.
 *
 * @author richwill28
 */
class Point {
    private double x;
    private double y;

    public Point(double x, double y) {
        this.x = x;
        this.y = y;
    }

    public double distanceTo(Point otherpoint) {
        double dx = this.x - otherpoint.x;
        double dy = this.y - otherpoint.y;
        return Math.sqrt(dx * dx + dy * dy);
    }

    @Override
    public String toString() {
        return "(" + this.x + ", " + this.y + ")";
    }
}
