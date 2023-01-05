#include "dheap.h"
#include "part2Submission.hpp"

void heapTest1();
void heapTest2();
void heapTest3();

int main() {
  heapTest1();
  heapTest2();
  heapTest3();
}

void heapTest1() {
  cout << "Heap Test 1" << endl;
  DHeap<char> ih(3);
  for (int i = 1; i < 5; i++) {
    char x = (char)'a' + i;
    ih.insert(x);
    cout << "Insert " << x << " into the heap." << endl;
    cout << "Heap Array:";
    ih.printHeapArray();
    cout << endl;
  }
}

void heapTest2() {
  cout << "Heap Test 2" << endl;
  DHeap<int> ih(3);
  for (int i = 1; i < 16; i++) {
    ih.insert(i);
    cout << "Insert " << i << " into the heap." << endl;
    cout << "Heap Array:";
    ih.printHeapArray();
    cout << endl;
  }
}

void heapTest3() {
  cout << "Heap Test 3" << endl;
  DHeap<int> ih(3);
  for (int i = 1; i < 16; i++) {
    ih.insert(i);
  }
  cout << "The heap from Test 2" << endl;
  ih.printHeapArray();
  cout << "Extract Max: " << ih.extractMax() << endl;
  ih.printHeapArray();
  cout << "Extract Max: " << ih.extractMax() << endl;
  ih.printHeapArray();
  cout << "Extract Max: " << ih.extractMax() << endl;
  ih.printHeapArray();
}
