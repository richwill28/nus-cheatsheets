#include <iostream>
#include "simpleIntLinkedList.h"
using namespace std;

ListNode::ListNode(int n) {
	_item = n;
	_next = NULL;
}

void List::insertHead(int n) {
	ListNode *aNewNode = new ListNode(n);
	aNewNode->_next = _head;
	_head = aNewNode;
	_size++;
}

void List::removeHead() {
	if (_size > 0) {
		ListNode *temp = _head;
		_head = _head->_next;
		delete temp;
		_size--;
	}
}

void List::print() {
    ListNode *temp = _head;
	while (temp != NULL) {
		cout << temp->_item << " ";
		temp = temp->_next;
	}
	cout << endl;
}

bool List::exist(int n) {
    ListNode *temp = _head;
	while (temp != NULL) {
		if (temp->_item == n) {
			return 1;
		}
		temp = temp->_next;
	}
	return 0;
}

int List::headItem() {
	if (_size > 0) {
		return _head->_item;
	}
	return -1;
}

bool List::empty() {
	return _size == 0;
}

int List::tailItem() {
	if (_size > 0) {
		ListNode *temp = _head;
		while (temp->_next != NULL) {
			temp = temp->_next;
		}
		return temp->_item;
	}
	return -1;
}

void List::removeTail() {
	if (_size == 1) {
		ListNode *temp = _head;
		_head = _head->_next;
		delete temp;
		_size--;
	} else if (_size > 1) {
		ListNode *prev = _head;
		ListNode *curr = _head->_next;
		while (curr->_next != NULL) {
			prev = curr;
			curr = curr->_next;
		}
		prev->_next = NULL;
		delete curr;
		_size--;
	}
}

List::~List() {
	while (_size != 0)
		removeHead();
}
