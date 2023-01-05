public class Main {
    public static void printObjects(Printable[] items) {
        for (Printable p : items) {
            p.print();
        }
    }

    public static void main(String[] args) {
        Printable[] printableItems = new Printable[] {
                new Circle(5),
                new Rectangle(3, 4),
                new Person("James Cook")
        };

        printObjects(printableItems);
    }
}
