#ifndef SIMPLELINKEDLISTTEMPLATEHPP
#define SIMPLELINKEDLISTTEMPLATEHPP

#include "simpleLinkedListTemplate.h"
#include <iostream>

using namespace std;

template <class T> ListNode<T>::ListNode(T n) {
  _item = n;
  _next = NULL;
}

template <class T> void List<T>::insertHead(T n) {
  ListNode<T> *aNewNode = new ListNode<T>(n);
  if (_head == NULL)
    _tail = aNewNode;
  aNewNode->_next = _head;
  _head = aNewNode;
  _size++;
};

template <class T> void List<T>::removeHead() {
  assert(_size > 0);
  if (_size > 0) {
    ListNode<T> *temp = _head;
    _head = _head->_next;
    delete temp;
    _size--;
    if (_size == 0)
      _tail = NULL;
  }
}

template <class T> void List<T>::print(bool withoutSpace) {

  ListNode<T> *temp = _head;
  while (temp) {
    cout << temp->_item;
    if (!withoutSpace)
      cout << " ";
    temp = temp->_next;
  }
  cout << endl;
}

template <class T> T List<T>::headItem() {
  assert(_head);
  return _head->_item;
}

template <class T> List<T>::~List() {
  while (_head)
    removeHead();
};

#endif
