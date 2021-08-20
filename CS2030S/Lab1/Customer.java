class Customer {
    private static int numOfCustomers = 0;
    private final int id;
    private final double arrivalTime;
    private final double serviceTime;

    public Customer(double arrivalTime, double serviceTime) {
        this.id = numOfCustomers++;
        this.arrivalTime = arrivalTime;
        this.serviceTime = serviceTime;
    }

    public int getId() {
        return id;
    }

    public double getArrivalTime() {
        return arrivalTime;
    }

    public double getServiceTime() {
        return serviceTime;
    }
}
