class ServiceBeginEvent extends Event {
    private Customer customer;
    private Counter counter;
    private Shop shop;

    public ServiceBeginEvent(double serviceBeginTime, Customer customer, Counter counter, Shop shop) {
        super(serviceBeginTime);
        this.customer = customer;
        this.counter = counter;
        this.shop = shop;
    }

    @Override
    public String toString() {
        return super.toString() + ": " + customer + " service begin (by " + counter + ")";
    }

    @Override
    public Event[] simulate() {
        counter.setAvailable(false);
        double serviceEndTime = getTime() + customer.getServiceDuration();
        return new Event[] {
                new ServiceEndEvent(serviceEndTime, customer, counter, shop)
        };
    }
}
