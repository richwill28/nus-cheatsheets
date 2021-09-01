class ServiceEnd extends Event {
    private Customer customer;
    private Counter counter;

    public ServiceEnd(double time, Customer customer, Counter counter) {
        super(time);
        this.customer = customer;
        this.counter = counter;
    }

    @Override
    public String toString() {
        return super.toString() + ": Customer " + this.customer.getId() + " " + "service done (by Counter " +
                this.counter.getId() + ")";
    }

    @Override
    public Event[] simulate() {
        this.counter.setAvailable(true);
        return new Event[] {new Departure(this.getTime(), this.customer)};
    }
}
