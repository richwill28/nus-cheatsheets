public class Arrival extends Event {
    private Customer customer;
    private Shop shop;

    public Arrival(double time, Customer customer, Shop shop) {
        super(time);
        this.customer = customer;
        this.shop = shop;
    }

    @Override
    public String toString() {
        return super.toString() + ": Customer " + this.customer.getId() + " " + "arrives";
    }

    @Override
    public Event[] simulate() {
        Counter counter = this.shop.getAvailableCounter();
        if (counter == null) {
            return new Event[] {new Departure(this.getTime(), this.customer)};
        } else {
            return new Event[] {new ServiceBegin(this.getTime(), this.customer, counter)};
        }
    }
}
