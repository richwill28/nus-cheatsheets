import java.lang.Math;

public class ShiftRegister implements ILFShiftRegister {
    private int[] register;
    private int tapIndex;

    public ShiftRegister(int size, int tap) {
        register = new int[size];
        tapIndex = size - (tap + 1);
    }

    @Override
    public void setSeed(int[] seed) {
        try {
            if (seed.length != register.length) {
                throw new IllegalArgumentException("Error: SeedMismatchException");
            }
            for (int i = 1; i <= seed.length; i++) {
                register[register.length - i] = seed[i - 1];
            }
        } catch(IllegalArgumentException e) {
            // Do nothing
            System.out.println(e.getMessage());
        }
    }

    @Override
    public int shift() {
        int leastSignificantBit = register[0] ^ register[tapIndex];
        System.arraycopy(register, 1, register, 0, register.length - 1);
        register[register.length - 1] = leastSignificantBit;
        return leastSignificantBit;
    }

    @Override
    public int generate(int k) {
        int extractedValue = 0;
        for (int i = k - 1; i >= 0; i--) {
            extractedValue += shift() * Math.pow(2, i);
        }
        return extractedValue;
    }

    public String getBit() {
        return this.toString();
    }

    @Override
    public String toString() {
        StringBuilder str = new StringBuilder();
        for (int bit : register) {
            str.append(bit);
        }
        return str.toString();
    }
}
