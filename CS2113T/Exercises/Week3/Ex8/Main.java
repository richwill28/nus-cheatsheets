public class Main {
    public static void printPrice(String item) {
        String name = item.substring(0, item.indexOf("--")).trim().toUpperCase();
        String price = item.substring(item.indexOf('$') + 1).replace('/', '.');
        System.out.println(name + ": " + price);
    }
    
    public static void main(String[] args) {
        printPrice("sandwich  --$4/50");
        printPrice("  soda --$10/00");
        printPrice("  fries --$0/50");
    }
}
