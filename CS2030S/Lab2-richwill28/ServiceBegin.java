class ServiceBegin extends Event {
    private Customer customer;
    private Counter counter;

    public ServiceBegin(double time, Customer customer, Counter counter) {
        super(time);
        this.customer = customer;
        this.counter = counter;
    }

    @Override
    public String toString() {
        return super.toString() + ": Customer " + this.customer.getId() + " " + "service begin (by Counter " +
                this.counter.getId() + ")";
    }

    @Override
    public Event[] simulate() {
        return new Event[] {
                new ServiceEnd(this.getTime() + customer.getServiceTime(), this.customer, this.counter)};
    }
}
