public class Main {
    public static void describe(String color, Priority p) {
        switch (p) {
        case LOW:
            System.out.println(color + " indicates low priority");
            break;
        case MEDIUM:
            System.out.println(color + " indicates medium priority");
            break;
        case HIGH:
            System.out.println(color + " indicates high priority");
            break;
        }
    }

    public static void main(String[] args) {
        describe("Red", Priority.HIGH);
        describe("Orange", Priority.MEDIUM);
        describe("Blue", Priority.MEDIUM);
        describe("Green", Priority.LOW);
    }
}
