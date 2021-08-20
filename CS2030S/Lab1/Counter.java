class Counter {
    private static int numOfCounters = 0;
    private final int id;
    private boolean available;

    public Counter() {
        this.id = numOfCounters++;
        this.available = true;
    }

    public int getId() {
        return this.id;
    }

    public boolean isAvailable() {
        return this.available;
    }

    public void setAvailable(boolean available) {
        this.available = available;
    }
}
