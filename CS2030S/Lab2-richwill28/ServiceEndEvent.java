class ServiceEndEvent extends Event {
    private Customer customer;
    private Counter counter;
    private Shop shop;

    public ServiceEndEvent(double serviceEndTime, Customer customer, Counter counter, Shop shop) {
        super(serviceEndTime);
        this.customer = customer;
        this.counter = counter;
        this.shop = shop;
    }

    @Override
    public String toString() {
        return super.toString() + ": " + customer + " service done (by " + counter + ")";
    }

    @Override
    public Event[] simulate() {
        counter.setAvailable(true);

        Customer nextCustomerInQueue = shop.getNextCustomerInQueue();
        if (nextCustomerInQueue == null) {
            return new Event[] {new DepartureEvent(getTime(), customer)};
        } else {
            return new Event[] {
                    new DepartureEvent(getTime(), customer),
                    new ServiceBeginEvent(getTime(), nextCustomerInQueue, counter, shop)
            };
        }
    }
}
