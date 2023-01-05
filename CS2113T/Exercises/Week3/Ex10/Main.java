import java.util.Scanner;
import java.util.Arrays;

public class Main {
    public static final double CONVERSION_RATE = 1.7;

    public static void findTotalExpenditure(String line) {
        String[] words = line.split(" ");
        String[] overseasExpenditure = new String[]{};
        double totalExpenditure = 0;
        for (String word : words) {
            if (word.indexOf('$') == 0) {
                totalExpenditure += Double.parseDouble(word.substring(1));
                overseasExpenditure = Arrays.copyOf(overseasExpenditure, overseasExpenditure.length + 1);
                overseasExpenditure[overseasExpenditure.length - 1] = word;
            }
        }
        double localExpenditure = totalExpenditure * CONVERSION_RATE;
        System.out.println("Expenses in overseas currency:" + Arrays.toString(overseasExpenditure));
        System.out.printf("Total in local currency: $%.2f\n", localExpenditure);
    }

    public static void main(String[] args) {
        String line;
        Scanner in = new Scanner(System.in);

        System.out.print("Your expenses while overseas?");
        line = in.nextLine();
        findTotalExpenditure(line);
    }
}
