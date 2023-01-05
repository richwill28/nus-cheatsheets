#pragma once
#include <math.h>

#include <iostream>
using namespace std;

#ifndef DHEAPHPP
#define DHEAPHPP

template <class T> void DHeap<T>::printHeapArray() {
  for (int i = 0; i < _n; i++) {
    cout << _heap[i] << " ";
  }
  cout << endl;
}

template <class T> int DHeap<T>::_lookFor(T x) {
  int i;
  for (i = 0; i < _n; i++) {
    if (_heap[i] == x) {
      return i;
    }
  }
  return -1;
}

#endif
