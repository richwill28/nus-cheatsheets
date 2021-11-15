/**
 * An implementation of the factorial function with recursive process.
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
 * An implementation of the factorial function with iterative process.
 * Time compplexity: O(n)
 * Space complexity: O(1)
 * 
 * @param {int} n The number.
 * @returns The factorial of n.
 */
function factorialWithIterativeProcess(n) {
    function iter(product, counter, maxCount) {
        return counter > maxCount ? product : iter(counter * product, counter + 1, maxCount);
    }
    return iter(1, 1, n);
}
