/**
 * An implementation of the summation formula with
 * recursive process.
 * 
 * @param {double} term The function.
 * @param {double} a Lower bound.
 * @param {double} next The function to compute the next index.
 * @param {double} b Upper bound.
 * @returns term(a) + term(a + 1) + ... + term(b).
 */
function sum(term, a, next, b) {
    return a > b ? 0 : term(a) + sum(term, next(a), next, b);
}

/**
 * An implementation of the summation formula with
 * iterative process.
 * 
 * @param {double} term The function.
 * @param {double} a Lower bound.
 * @param {double} next The function to compute the next index.
 * @param {double} b Upper bound.
 * @returns term(a) + term(a + 1) + ... + term(b).
 */
function iterativeSum(term, a, next, b) {
    function iter(a, result) {
        return a > b ? result : iter(next(a), result + term(a));
    }
    return iter(a, 0);
}

/**
 * Compute 1 x 2 + 2 x 3 + ... + n x (n + 1).
 * Time complexity: O(n)
 * Space complexity: O(n)
 * 
 * @param {int} n An integer.
 * @returns 1 x 2 + 2 x 3 + ... + n x (n + 1).
 */
function mySumWithoutHigherOrder(n) {
    return n < 1 ? 0 : n * (n + 1) + mySum(n - 1);
}

/**
 * Compute 1 x 2 + 2 x 3 + ... + n x (n + 1).
 * 
 * @param {int} n An integer.
 * @returns 1 x 2 + 2 x 3 + ... + n x (n + 1).
 */
function mySumWithHigherOrder(n) {
    return sum(x => x * (x + 1), 1, x => x + 1, n);
}

/**
 * Calculate the definite integral of a function between
 * the limits a and b. The result of this intgration is
 * approximated using the summation formula, and is only
 * valid for small value of dx.
 * 
 * @param {double} f The function.
 * @param {double} a Lower bound of the limit.
 * @param {double} b Upper bound of the limit.
 * @param {double} dx The interval size.
 * @returns The definite integral of f between the limits
 *          a and b.
 */
function integral(f, a, b, dx) {
    return sum(f, a + dx / 2, x => x + dx, b) * dx;
}

/**
 * Calculate the definite integral of a function between
 * the limits a and b. The result of this integration is
 * derived from Simpson's Rule.
 * 
 * @param {double} f The function.
 * @param {double} a Lower bound of the limit.
 * @param {double} b Upper bound of the limit.
 * @param {double} n The number of intervals.
 * @returns 
 */
function simpsonRule(f, a, b, n) {
    const h = (b - a) / n;
    const next = x => x + 2 * h;
    return h / 3 * (f(a) + 4 * sum(f, a + h, next, a + n * h) + 2 * sum(f, a + 2 * h, next, a + n * h) - (n % 2 ? 3 * f(a + n * h) : f(a + n * h)));
}
