import java.util.Arrays;

public class Main {
    public static String[] filterEmails(String[] items) {
        String[] emails = new String[]{};
        for (String item : items) {
            if (item.indexOf('@') != -1) {
                emails = Arrays.copyOf(emails, emails.length + 1);
                emails[emails.length - 1] = item;
            }
        }
        return emails;
    }

    public static void printItems(String[] items) {
        System.out.println(Arrays.toString(items));
    }

    public static void main(String[] args) {
        printItems(filterEmails(new String[]{}));
        printItems(filterEmails(new String[]{"abc"}));
        printItems(filterEmails(new String[]{"adam@example.com", "aab", "john@example.com", "some@"}));
        printItems(filterEmails(new String[]{"xyz", "@bee.com", "aab"}));
    }
}
