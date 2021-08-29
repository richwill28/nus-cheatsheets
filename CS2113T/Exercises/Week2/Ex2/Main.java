class Main {
    public static void main(String[] args) {
        String first = args[0];
        String second = args[1];
        boolean isSame = first.equals(second);
        System.out.println("Words given: " + first + ", " + second);
        System.out.println("They are the same: " + isSame);
    }
}
