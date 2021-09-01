class Shop {
    private Counter[] counters;
    private Queue queue;

    public Shop(int numOfCounters, int maxQueueLength) {
        this.counters = new Counter[numOfCounters];
        for (int i = 0; i < numOfCounters; i++) {
            counters[i] = new Counter();
        }
        this.queue = new Queue(maxQueueLength);

    }

    public Counter getAvailableCounter() {
        Counter counter = null;
        for (int i = 0; i < counters.length; i++) {
            if (counters[i].isAvailable()) {
                counter = counters[i];
                counters[i].setAvailable(false);
                break;
            }
        }
        return counter;
    }

    public void addToQueue(Customer customer) {
        queue.enq(customer);
    }

    public Customer getNextCustomerInQueue() {
        return (Customer) queue.deq();
    }

    public boolean isQueueFull() {
        return queue.isFull();
    }

    public boolean isQueueEmpty() {
        return queue.isEmpty();
    }

    public String getQueue() {
        return queue.toString();
    }
}
