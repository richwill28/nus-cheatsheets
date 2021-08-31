import java.awt.Rectangle;

public class Main {
    public static void main(String[] args) {
        Rectangle r = new Rectangle(0, 0, 4, 6);
        System.out.println(r);

        int area = r.width * r.height;
        System.out.println("Area: " + area);

        r.setsSize(8, 10);
        System.out.println(r);
    }
}
