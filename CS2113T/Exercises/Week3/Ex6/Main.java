public class Main {
    public static void main(String[] args) {
        Circle c = new Circle();
        System.out.println("max radius used so far : " + Circle.getMaxRadius());
        c = new Circle(0, 0, 10);
        System.out.println("max radius used so far : " + Circle.getMaxRadius());
        c = new Circle(0, 0, -15);
        System.out.println("max radius used so far : " + Circle.getMaxRadius());
        c.setRadius(12);
        System.out.println("max radius used so far : " + Circle.getMaxRadius());
    }
}
