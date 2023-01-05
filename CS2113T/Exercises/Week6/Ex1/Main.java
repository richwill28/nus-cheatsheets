import java.util.ArrayList;

public class Main {
    private static ArrayList<Integer> numbers = new ArrayList<>();

    private static void addNumber(int i) {
        numbers.add(Integer.valueOf(i));
        System.out.println(numbers);
    }

    private static int getTotal() {
        int sum = 0;
        for (Integer number : numbers) {
            sum += number.intValue();
        }
        return sum;
    }

    private static boolean isFound(int i) {
        return numbers.contains(Integer.valueOf(i));
    }

    private static void removeNumber(int i) {
        numbers.remove(Integer.valueOf(i));
        System.out.println(numbers);
    }

    public static void main(String[] args) {
        System.out.println("Adding numbers to the list");
        addNumber(3);
        addNumber(8);
        addNumber(24);
        System.out.println("The total is: " + getTotal());
        System.out.println("8 in the list : " + isFound(8) );
        System.out.println("5 in the list : " + isFound(5) );
        removeNumber(8);
        System.out.println("The total is: " + getTotal());
    }
}
