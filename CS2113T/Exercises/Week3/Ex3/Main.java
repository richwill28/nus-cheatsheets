import java.awt.Point;
import java.awt.Rectangle;

public class Main {
    public static Point move(Point p, Rectangle r) {
        if (p == null || r == null) {
            return null;
        }

        int rx = r.x;
        int ry = r.y;
        r.x = p.x;
        r.y = p.y;
        return new Point(rx, ry);
    }
    
    public static void main(String[] args) {
        Point p1 = new Point(0, 0);
        Rectangle r1 = new Rectangle(2, 3, 5, 6);
        System.out.println("arguments: " + p1 + ", " + r1);
        Point p2 = move(p1, r1);
        System.out.println("argument point after method call: " + p1);
        System.out.println("argument rectangle after method call: " + r1);
        System.out.println("returned point: " + p2);
        System.out.println(move(null, null));
    }
}
