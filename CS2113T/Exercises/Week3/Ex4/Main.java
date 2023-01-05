public class Main {
    public static void main(String[] args) {
        Circle c = new Circle();
        System.out.println(c.getArea());
        c = new Circle(1, 2, 5);
        System.out.println(c.getArea());
    }
}
