public class Main {
    /**
     * Computes argA to the power of argB.
     *
     * @param argA The first number.
     * @param argB The second number.
     * @return argA to the power of argB.
     */
    static int MysteryFunction(int argA, int argB) {
        int c = 1;
        int d = argA;
        int e = argB;
        while (e > 0) {
            if (2 * (e / 2) != e) {
                c = c * d;
            }
            d = d * d;
            e = e / 2;
        }
        return c;
    }

    public static void main(String[] args) {
        int output = MysteryFunction(5, 5);
        System.out.println("The answer is: " + output + ".");
    }
}
