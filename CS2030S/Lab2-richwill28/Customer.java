class Customer {
    private static int numOfCustomers = 0;
    private final int id;
    private final double arrivalTime;
    private final double serviceDuration;

    public Customer(double arrivalTime, double serviceDuration) {
        this.id = numOfCustomers++;
        this.arrivalTime = arrivalTime;
        this.serviceDuration = serviceDuration;
    }

    public double getArrivalTime() {
        return arrivalTime;
    }

    public double getServiceDuration() {
        return serviceDuration;
    }

    @Override
    public String toString() {
        return "C" + id;
    }
}
