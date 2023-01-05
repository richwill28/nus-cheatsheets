#pragma once
using namespace std;

template <class T> class List;

template <class T> class ListNode {
private:
  T _item;
  ListNode<T> *_next;

public:
  ListNode(T);
  T content() { return _item; }
  void print() { cout << _item; }
  friend class List<T>;
};

template <class T> class List {
private:
  int _size;
  ListNode<T> *_head;

public:
  List() {
    _size = 0;
    _head = NULL;
  }
  ~List();
  void insertHead(T);
  void print();
  int size() { return _size; }
  bool empty() { return _size == 0; }
  T headItem();
  void removeHead();
  T &operator[](int idx);
  void insertAtPos(int idx, T item);
  void removeAtPos(int idx);
};

#include "simpleLinkedListTemplate.hpp"
