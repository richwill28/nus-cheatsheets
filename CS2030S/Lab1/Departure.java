public class Departure extends Event {
    private Customer customer;

    public Departure(double time, Customer customer) {
        super(time);
        this.customer = customer;
    }

    @Override
    public String toString() {
        return super.toString() + ": Customer " + this.customer.getId() + " " + "departed";
    }

    @Override
    public Event[] simulate() {
        return new Event[0];
    }
}
