class PrimalityTest {
    /**
     * Determine whether n is divisible by d.
     * 
     * @param {int} n The number to be divided (dividend).
     * @param {int} d The divisor.
     * @returns True if n is divisible by d, otherwise false.
     */
    static #isDivisible(n, d) {
        return n % d ? false : true;
    }

    /**
     * Generate a uniformly-disributed random integer
     * between [min, max).
     * 
     * @param {int} min The minimum number (inclusive).
     * @param {int} max The maximum number (exclusive).
     */
    static #randomNumberGenerator(min, max) {
        min = Math.ceil(min);
        max = Math.floor(max);
        return Math.floor(Math.random() * (max - min) + min);
    }

    /**
     * Determine whether n is a prime number. The algorithm
     * used in this function is based on the fact that if n
     * is not prime it must have a divisor less than or equal
     * to sqrt(n).
     * Time complexity: O(sqrt(n))
     * 
     * @param {int} n The number to be tested.
     * @returns True if n is prime, otherwise false.
     */
    static isPrime(n) {
        if (n <= 1) {
            return false;
        }

        for (let i = 2; i <= Math.sqrt(n); i++) {
            if (this.#isDivisible(n, i)) {
                return false;
            }
        }
        return true;
    }

    /**
     * Determine whether n is a prime number. The algorithm used
     * in this function is based on a result from a number theory
     * known as Fermat's Little Theorem. Hence, this function is
     * probabilistic and the assertion on the primality of n is
     * not always correct. In particular, numbers that can fool
     * this function are called Carmichael numbers, but they are
     * extremely rare.
     * Time complexity: O(t log n)
     * 
     * @param {int} n The number to be tested.
     * @param {int} t The number of trial to test n.
     * @returns True if n is prime, otherwise false.
     */
    static fermatTest(n, t) {
        if (n <= 1) {
            return false;
        }

        const a = this.#randomNumberGenerator(2, n);
        for (let i = 0; i < t; i++) {
            // to be replaced with better implementation
            if (Math.pow(a, n) % n !== a) {
                return false;
            }
        }
        return true;
    }
}
