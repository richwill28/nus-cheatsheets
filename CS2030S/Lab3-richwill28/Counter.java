class Counter {
    private static int numOfCounters = 0;
    private final int id;
    private boolean isAvailable;

    public Counter() {
        this.id = numOfCounters++;
        this.isAvailable = true;
    }

    public boolean isAvailable() {
        return isAvailable;
    }

    public void setAvailable(boolean isAvailable) {
        this.isAvailable = isAvailable;
    }

    @Override
    public String toString() {
        return "S" + id;
    }
}
