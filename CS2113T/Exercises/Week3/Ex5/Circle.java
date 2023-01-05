public class Circle {
    private int x;
    private int y;
    private double radius;

    public Circle() {
        this(0, 0, 0);
    }

    public Circle(int x, int y, double radius) {
        this.x = x;
        this.y = y;
        if (radius < 0) {
            radius = 0;
        }
        this.radius = radius;
    }

    public int getX() {
        return x;
    }

    public int getY() {
        return y;
    }

    public double getRadius() {
        return radius;
    }

    public int getArea() {
        double area = Math.PI * Math.pow(radius, 2);
        return (int)area;
    }

    public void setX(int x) {
        this.x = x;
    }

    public void setY(int y) {
        this.y = y;
    }

    public void setRadius(double radius) {
        if (radius < 0) {
            radius = 0;
        }
        this.radius = radius;
    }
}
