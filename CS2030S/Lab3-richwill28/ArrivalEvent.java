class ArrivalEvent extends Event {
    private Customer customer;
    private Shop shop;

    public ArrivalEvent(double arrivalTime, Customer customer, Shop shop) {
        super(arrivalTime);
        this.customer = customer;
        this.shop = shop;
    }

    @Override
    public String toString() {
        return super.toString() + ": " + customer + " arrived " + shop.getQueue();
    }

    @Override
    public Event[] simulate() {
        Counter counter = shop.getAvailableCounter();
        if (counter == null) {
            if (shop.isQueueFull()) {
                return new Event[] {
                        new DepartureEvent(getTime(), customer)
                };
            } else {
                return new Event[] {
                        new JoinQueueEvent(getTime(), customer, shop)
                };
            }
        } else {
            return new Event[] {
                    new ServiceBeginEvent(getTime(), customer, counter, shop)
            };
        }
    }
}
