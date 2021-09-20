#pragma once

#ifndef SIMPLELINKEDLISTTEMPLATEHPP
#define SIMPLELINKEDLISTTEMPLATEHPP

#include <iostream>
#include "simpleLinkedListTemplate.h"
using namespace std;

template <class T>
ListNode<T>::ListNode(T n) {
	_item = n;
	_next = NULL;
}

template <class T>
void List<T>::insertHead(T n) {
	ListNode<T> *newNode = new ListNode<T>(n);
	newNode->_next = _head;
	_head = newNode;
	_size++;
}

template <class T>
void List<T>::removeHead() {
	if (_size > 0) {
		ListNode<T> *temp = _head;
		_head = _head->_next;
		delete temp;
		_size--;
	}
}

template <class T>
void List<T>::print(bool withNL) {
	ListNode<T> *temp = _head;
	while (temp) {
		cout << temp->_item;
		if (withNL)
			cout << endl;
		else
			cout << " ";
		temp = temp->_next;
	}
	cout << endl;
}

template <class T>
bool List<T>::exist(T n) {
	ListNode<T> *temp = _head;
	while (temp != NULL) {
		if (temp->_item == n) {
			return true;
		}
		temp = temp->_next;
	}
	return false;
}

template <class T>
T List<T>::headItem() {
	if (_size > 0) {
		return _head->_item;
	}
	return -1;
}

template <class T>
T List<T>::extractMax() {
	ListNode<T> *prevNode = NULL;
	ListNode<T> *maxNode = _head;
	ListNode<T> *currNode = _head;
	while (currNode->_next != NULL) {
		if (currNode->_next->_item > maxNode->_item) {
			maxNode = currNode->_next;
			prevNode = currNode;
		}
		currNode = currNode->_next;
	}
	T maxItem = maxNode->_item;
	if (maxNode == _head) {
		removeHead();
	} else {
		prevNode->_next = maxNode->_next;
		delete maxNode;
		_size--;
	}
	return maxItem;
}

template <class T>
void List<T>::reverseOp() {
    ListNode<T> *prevNode = NULL;
	ListNode<T> *currNode = _head;
	ListNode<T> *nextNode = NULL;
	while (currNode != NULL) {
		nextNode = currNode->_next;
		currNode->_next = prevNode;
		prevNode = currNode;
		currNode = nextNode;
	}
	_head = prevNode;
}

template <class T>
List<T>::~List() {
	while (_head) {
		removeHead();
	}
}

#endif
