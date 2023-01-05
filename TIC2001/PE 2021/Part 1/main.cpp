#include "part1Submission.hpp"
#include "simpleLinkedListTemplate.h"
#include <iostream>

using namespace std;

void testOperator1();
void testInsertAtPos();
void testRemoveAtPos();
void testMixed();

int main() {
  testOperator1();
  testInsertAtPos();
  testRemoveAtPos();
  testMixed();
  return 0;
}

void testOperator1() {
  cout << "Operator [] Test" << endl;
  List<char> lc;

  lc.insertHead('a');
  lc.insertHead('b');
  lc.insertHead('c');
  lc.insertHead('d');
  lc.insertHead('e');
  lc.insertHead('f');
  lc.insertHead('1');
  lc.insertHead('2');
  lc.insertHead('3');
  lc.insertHead('4');
  lc.insertHead('5');
  lc.insertHead('6');

  cout << "The linked list so far: ";
  lc.print();
  for (int i = 0; i < 10; i += 2) {
    cout << "The char with indes " << i << " is " << lc[i] << endl;
  }
}

void testInsertAtPos() {
  cout << "insertAtPos Test" << endl;
  List<char> lc;

  lc.insertHead('a');
  lc.insertHead('b');
  lc.insertHead('c');
  lc.insertHead('d');
  lc.insertHead('e');
  lc.insertHead('f');

  lc.insertAtPos(4, '#');
  lc.insertAtPos(6, '#');
  lc.insertAtPos(8, '+');
  lc.print();
  lc.insertAtPos(10, '*');
  lc.insertAtPos(0, '>');
  lc.print();
  for (int i = 0; i < 9; i += 3) {
    cout << "The char with indes " << i << " is " << lc[i] << endl;
  }
}

void testRemoveAtPos() {
  cout << "RemoveAtPos Test" << endl;
  List<char> lc;

  lc.insertHead('a');
  lc.insertHead('b');
  lc.insertHead('c');
  lc.insertHead('d');
  lc.insertHead('e');
  lc.insertHead('f');
  lc.insertHead('1');
  lc.insertHead('2');
  lc.insertHead('3');
  lc.insertHead('4');
  lc.insertHead('5');
  lc.insertHead('6');

  cout << "The linked list so far: ";
  lc.print();
  lc.removeAtPos(11);
  lc.removeAtPos(7);
  lc.removeAtPos(3);
  lc.removeAtPos(0);
  lc.print();
  for (int i = 0; i < 7; i += 2) {
    cout << "The char with indes " << i << " is " << lc[i] << endl;
  }
}

void testMixed() {
  cout << "Mixed Test" << endl;
  List<int> li;

  for (int i = 0; i <= 100; i += 10) {
    li.insertHead(i);
  }

  li.removeHead();
  li.print();
  li.removeAtPos(4);
  li.removeAtPos(2);
  li.insertAtPos(7, 999);
  li.insertAtPos(9, 998);
  li.insertAtPos(0, 996);
  li.print();
}
