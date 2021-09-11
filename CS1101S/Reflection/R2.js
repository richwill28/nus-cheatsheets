/**
 * An implementation of the factorial function
 * with recursive process.
 * Time complexity: O(n)
 * Space complexity: O(n)
 * 
 * @param {int} n The number.
 * @returns The factorial of n.
 */
function factorialWithRecursiveProcess(n) {
    return n === 1 ? 1 : n * factorialWithRecursiveProcess(n - 1);
}

/**
 * An implementation of the factorial function
 * with iterative process.
 * Time compplexity: O(n)
 * Space complexity: O(1)
 * 
 * @param {int} n The number.
 * @returns The factorial of n.
 */
function factorialWithIterativeProcess(n) {
    /**
     * Calculate the factorial of n with iterative process.
     * 
     * @param {int} product The product of n factorial.
     * @param {int} counter The current iteration.
     * @param {int} maxCount The maximum number of iteration.
     * @returns 
     */
    function factorialIterator(product, counter, maxCount) {
        return counter > maxCount ? product : factorialIterator(counter * product, counter + 1, maxCount);
    }

    return factorialIterator(1, 1, n);
}
