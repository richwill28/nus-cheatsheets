class Shop {
    private Counter[] counters;

    public Shop(int numOfCounters) {
        this.counters = new Counter[numOfCounters];
        for (int i = 0; i < numOfCounters; i++) {
            counters[i] = new Counter();
        }
    }

    public Counter getAvailableCounter() {
        Counter counter = null;
        for (int i = 0; i < counters.length; i++) {
            if (counters[i].isAvailable()) {
                counter = counters[i];
                counter.setAvailable(false);
                break;
            }
        }
        return counter;
    }
}
