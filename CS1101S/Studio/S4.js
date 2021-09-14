function compose(f, g) {
    return x => f(g(x));
}

// equivalent to x => f(f(x))
function twice(f) {
    return compose(f, f);
}

// equivalent to x => f(f(f(x)))
function thrice(f) {
    return compose(f, compose(f, f));
}

function repeated(f, n) {
    return n === 0
        ? x => x
        : compose(f, repeated(f, n - 1));
}

console.log(twice(thrice)(x => x + 1)(0));          // 9
console.log(thrice(thrice)(x => x + 1)(0));         // 27

const square = x => x * x;
const add1 = x => x + 1;

console.log(((thrice(thrice))(add1))(6));           // 33
console.log(((thrice(thrice))(x => x))(compose));   // [Function: compose]
console.log(((thrice(thrice))(square))(1));         // 1
console.log(((thrice(thrice))(square))(2));         // Infinity

/**
 * Return the element of the specified row and
 * position of Pascal's triangle.
 * 
 * @param {int} row The row of Pascal's triangle counted
 *                  from the top, starting with 1.
 * @param {int} position The position counted from the
 *                       left of the row, starting with 1.
 * @returns The element of the specified row and
 *          position of Pascal's triangle.
 */
function pascal(row, position) {
    return position === 1
        ? 1
        : row === position
            ? 1
            : pascal(row - 1, position - 1) + pascal(row - 1, position);
}

console.log(pascal(3, 2));      // 2
console.log(pascal(5, 3));      // 6
