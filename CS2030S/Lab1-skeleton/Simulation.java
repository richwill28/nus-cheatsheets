/**
 * This class is a general abstract class that
 * encapsulates a simulation.  To implement a
 * simulation, inherit from this class and implement
 * the `getInitialEvents` method.  
 *
 * @author Wei Tsang
 * @version CS2030S AY20/21 Semester 2
 */
abstract class Simulation {
  /**
   * An abstract method to return an array of events
   * used to initialize the simulation.
   *
   * @return An array of initial events that the 
   *         simulator can use to kick-start the 
   *         simulation.
   */
  public abstract Event[] getInitialEvents();
}
