import java.util.Scanner;

/**
 * This class implements a shop simulation.
 *
 * @author Wei Tsang
 * @version CS2030S AY20/21 Semester 2
 */
class ShopSimulation extends Simulation {
    private final Shop shop;

    /**
     * The list of customer arrival events to populate
     * the simulation with.
     */
    private Event[] initEvents;

    /**
     * Constructor for a shop simulation.
     *
     * @param sc A scanner to read the parameters from.  The first
     *           integer scanned is the number of customers; followed
     *           by the number of service counters.  Next is a
     *           sequence of (arrival time, service time) pair, each
     *           pair represents a customer.
     */
    public ShopSimulation(Scanner sc) {
        initEvents = new Event[sc.nextInt()];
        int numOfCounters = sc.nextInt();
        int maxQueueLength = sc.nextInt();
        shop = new Shop(numOfCounters, maxQueueLength);
        for (int i = 0; i < initEvents.length; i++) {
            double arrivalTime = sc.nextDouble();
            double serviceDuration = sc.nextDouble();
            Customer customer = new Customer(arrivalTime, serviceDuration);
            initEvents[i] = new ArrivalEvent(arrivalTime, customer, shop);
        }
    }

    /**
     * Retrieve an array of events to populate the
     * simulator with.
     *
     * @return An array of events for the simulator.
     */
    @Override
    public Event[] getInitialEvents() {
        return initEvents;
    }
}
