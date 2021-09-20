import static org.junit.Assert.*;

import org.junit.Test;

public class LoadBalancingTest {
    /* Tests for Problem 2 */

    @Test
    public void testFeasibleLoad1() {
        int[] jobs = {1, 3, 5, 7, 9, 11, 10, 8, 6, 4};
        int processors = 10;
        int queryLoad = 10;
        assertEquals(LoadBalancing.feasibleLoad(jobs, queryLoad, processors), false);
    }

    @Test
    public void testFeasibleLoad2() {
        int[] jobs = {1, 2, 3};
        int processors = 1;
        int queryLoad = 6;
        assertEquals(LoadBalancing.feasibleLoad(jobs, queryLoad, processors), true);
    }

    @Test
    public void testFindLoad1() {
        int[] jobSize = {1, 3, 5, 7, 9, 11, 10, 8, 6, 4};
        int processors = 5;
        assertEquals(LoadBalancing.findLoad(jobSize, processors), 18);
    }

    @Test
    public void testFindLoad2() {
        int[] jobSize = {67, 65, 43, 42, 23, 17, 9, 100};
        int processors = 3;
        assertEquals(LoadBalancing.findLoad(jobSize, processors), 132);
    }

    @Test
    public void testFindLoad3() {
        int[] jobSize = {4, 100, 80, 15, 20, 25, 30};
        int processors = 2;
        assertEquals(LoadBalancing.findLoad(jobSize, processors), 170);
    }

    @Test
    public void testFindLoad4() {
        int[] jobSize = {2, 3, 4, 5, 6, 7, 8, 100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83};
        int processors = 8;
        assertEquals(LoadBalancing.findLoad(jobSize, processors), 261);
    }

    @Test
    public void testFindLoad5() {
        int[] jobSize = {7};
        int processors = 1;
        assertEquals(LoadBalancing.findLoad(jobSize, processors), 7);
    }
}
