public interface ILFShiftRegister {
    public void setSeed(int[] seed);
    public int shift();
    public int generate(int k);
}
