class JoinQueueEvent extends Event {
    private Customer customer;
    private Shop shop;

    public JoinQueueEvent(double joinQueueTime, Customer customer, Shop shop) {
        super(joinQueueTime);
        this.customer = customer;
        this.shop = shop;
    }

    @Override
    public String toString() {
        return super.toString() + ": " + customer + " joined queue " + shop.getQueue();
    }

    @Override
    public Event[] simulate() {
        shop.addToQueue(customer);
        return new Event[0];
    }
}
