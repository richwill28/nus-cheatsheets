import java.util.PriorityQueue;

/**
 * This class implements a discrete event simulator.
 * The simulator maintains a priority queue of events.
 * It runs through the events and simulates each one until 
 * the queue is empty.
 *
 * @author Wei Tsang
 * @version CS2030S AY20/21 Semester 2
 */
public class Simulator {
  /** The event queue. */
  private final PriorityQueue<Event> events;

  /** 
   * The constructor for a simulator.  It takes in
   * a simulation as an argument, and calls the 
   * getInitialEvents method of that simulation to
   * initialize the event queue.
   *
   * @param simulation The simulation to simulate.
   */
  public Simulator(Simulation simulation) {
    this.events = new PriorityQueue<Event>();
    for (Event e : simulation.getInitialEvents()) {
      this.events.add(e);
    }
  }

  /**
   * Run the simulation until no more events is in
   * the queue.  For each event in the queue (in
   * increasing order of time), print out its string 
   * representation, then simulate it.  If the 
   * simulation returns one or more events, add them 
   * to the queue, and repeat.
   */
  public void run() {
    Event event = this.events.poll();
    while (event != null) {
      System.out.println(event);
      Event[] newEvents = event.simulate();
      for (Event e : newEvents) {
        this.events.add(e);
      }
      event = this.events.poll();
    }
    return;
  }
}
