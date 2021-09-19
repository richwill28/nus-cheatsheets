import static org.junit.Assert.*;

import org.junit.Test;

public class ShiftRegisterTest {
    /**
     * getRegister returns a shiftregister to test
     * @param size
     * @param tap
     * @return a new shift register
     * Description: to test a shiftregister, update this function
     * to instantiate the shift register
     */
    ILFShiftRegister getRegister(int size, int tap) {
        return new ShiftRegister(size, tap);
    }

    /**
     * Test shift with simple example
     */
    @Test
    public void testShift1() {
        ILFShiftRegister r = getRegister(9, 7);
        int[] seed = {0, 1, 0, 1, 1, 1, 1, 0, 1};
        r.setSeed(seed);
        int[] expected = {1, 1, 0, 0, 0, 1, 1, 1, 1, 0};
        for (int i = 0; i < 10; i++) {
            assertEquals(expected[i], r.shift());
        }
    }

    /**
     * Test generate with simple example
     */
    @Test
    public void testGenerate1() {
        ILFShiftRegister r = getRegister(9, 7);
        int[] seed = {0, 1, 0, 1, 1, 1, 1, 0, 1};
        r.setSeed(seed);
        int[] expected = {6, 1, 7, 2, 2, 1, 6, 6, 2, 3};
        for (int i = 0; i < 10; i++) {
            assertEquals("GenerateTest", expected[i], r.generate(3));
        }
    }

    /**
     * Test register of length 1
     */
    @Test
    public void testOneLength() {
        ILFShiftRegister r = getRegister(1, 0);
        int[] seed = {1};
        r.setSeed(seed);
        int[] expected = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (int i = 0; i < 10; i++) {
            assertEquals(expected[i], r.generate(3));
        }
    }

    /**
     * Test with erroneous seed
     */
    @Test
    public void testError() {
        ILFShiftRegister r = getRegister(4, 1);
        int[] seed = {1, 0, 0, 0, 1, 1, 0};
        r.setSeed(seed);
        r.shift();
        r.generate(4);
    }
}
