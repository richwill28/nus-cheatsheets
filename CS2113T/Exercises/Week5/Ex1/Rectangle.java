public class Rectangle extends Shape {
    private int height;
    private int width;

    public Rectangle(int height, int width) {
        this.height = height;
        this.width = width;
    }

    @Override
    public int area() {
        return height * width;
    }

    @Override
    public void print() {
        System.out.println("Rectangle of area " + area());
    }
}
