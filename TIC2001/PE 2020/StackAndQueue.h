#pragma once
#include "simpleLinkedListTemplate.h"

template <class T> class Stack {
private:
  List<T> _l;

public:
  void push(T item);
  T pop();
  void print(bool withoutSpace = false) { _l.print(withoutSpace); };
  bool empty() { return _l.size() == 0; };
};

template <class T> class Queue {
private:
  List<T> _l;

public:
  void enqueue(T item);
  T dequeue();
  void print(bool withoutSpace = false) { _l.print(withoutSpace); };
  bool empty() { return _l.size() == 0; };
};
