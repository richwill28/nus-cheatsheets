#include "BST.h"

void testInsertion1(bool printWithHeight = true);
void testInsertion2(bool printWithHeight = true);
void testSuccessor();
void testSearchMinMax();
void testExist();

int main() {
    testInsertion1(true);
    testSearchMinMax();
    testExist();
    testSuccessor();
    testInsertion2(true);
}

void testInsertion1(bool printWithHeight) {
    cout << "Insertion Test 1" << endl;
    int array[] = { 7, 3, 1, 0, 2, 5, 4, 6, 11, 9, 8, 10, 13, 12, 14 };
    BinarySearchTree<int> bsti;
    for (int i = 0; i < 15; i++) {
        bsti.insert(array[i]);
    }

    bsti.printTree(false);
    cout << endl << endl;
    bsti.printTree(printWithHeight);
    cout << endl << endl;
    cout << "The size of the tree is " << bsti.size() << endl;
    cout << "Pre-order Traversal:" << endl;
    bsti.preOrderPrint();
    cout << "In-order Traversal:" << endl;
    bsti.inOrderPrint();
    cout << "Post-order Traversal:" << endl;
    bsti.postOrderPrint();
	cout << endl << endl;
}

void testExist() {
    cout << "Exist Test" << endl;
    BinarySearchTree<int> bsti;
    cout << "Numbers inserted in the tree: ";
    for (int i = 0; i < 11; i++) {
        cout << i * 6 << " ";
        bsti.insert(i * 6);
    }

    // bsti.printTree(false);
    cout << endl << endl;

    for (int i = 0; i < 70; i += 8) {
        cout << "The number " << i << (bsti.exist(i) ? " exists " : " does not exist ") << "in the tree" << endl;
    }
    cout << endl << endl;
}

void testInsertion2(bool printWithHeight) {
    cout << "Insertion Test 2" << endl;
    cout << "The tree shape should be the same as Test 1" << endl;
    cout << "if you have done the balancing correctly." << endl;
    BinarySearchTree<int> bsti;
    for (int i = 0; i < 15; i++) {
        bsti.insert(i);
    }

    bsti.printTree(printWithHeight);
    cout << endl << endl;
    cout << "The size of the tree is " << bsti.size() << endl;
    cout << "Pre-order Traversal:" << endl;
    bsti.preOrderPrint();
    cout << "In-order Traversal:" << endl;
    bsti.inOrderPrint();
    cout << "Post-order Traversal:" << endl;
    bsti.postOrderPrint();
    cout << endl << endl;
}

void testSearchMinMax() {
    cout << "Search Min/Max Test" << endl;
    int array[] = { 7, 3, 1, 0, 2, 5, 4, 6, 11, 9, 8, 10, 13, 12, 14 };
    BinarySearchTree<int> bsti;
    for (int i = 0; i < 15; i++) {
        bsti.insert(array[i]);
    }

    cout << "The minimum number in the tree is " << bsti.searchMin() << endl;
    cout << "The maximum number in the tree is " << bsti.searchMax() << endl;
    cout << endl;
}

void testSuccessor() {
    cout << "Successor Test" << endl;
    BinarySearchTree<int> bsti;
    cout << "Numbers inserted in the tree: ";
    for (int i = 0; i < 11; i++) {
        cout << i * 7 << " ";
        bsti.insert(i * 7);
    }

    // bsti.printTree(false);
    cout << endl << endl;

    for (int i = 0; i < 70; i += 10) {
        cout << "The successor of " << i << " in the BST is " << bsti.successor(i) << endl;
    }
    cout << endl << endl;
}
