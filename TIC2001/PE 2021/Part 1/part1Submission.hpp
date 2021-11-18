#include <iostream>
using namespace std;

// if you use Mac, just in case your code doesn't work, do this:
// sudo xcode-select --switch /Library/Developer/CommandLineTools/

template <class T>
T& List<T>::operator[](int idx) {
    if (idx >= _size) {
        cout << "Index out of bound error (operator[])" << endl;
        exit(1);    // not the best practice, we often call this overreacting
    }

    ListNode<T>* temp = _head;
    for (int i = 0; i < idx; i++) {
        temp = temp->_next;
    }
    return temp->_item;
}

template <class T>
void List<T>::insertAtPos(int idx, T item) {
    if (idx < 0 || idx > _size) {
        cout << "Index out of bound error (insertAtPos)" << endl;
        return;
    }

    if (idx == 0) {
        insertHead(item);
    } else {
        ListNode<T>* temp = _head;
        for (int i = 1; i < idx; i++) {
            temp = temp->_next;
        }

        ListNode<T>* newNode = new ListNode<T>(item);
        newNode->_next = temp->_next;
        temp->_next = newNode;
        _size++;
    }
}

template <class T>
void List<T>::removeAtPos(int idx) {
    if (idx < 0 || idx >= _size) {
        cout << "Index out of bound error (removeAtPos)" << endl;
        return;
    }

    if (idx == 0) {
        removeHead();
    } else {
        ListNode<T>* temp = _head;
        for (int i = 1; i < idx; i++) {
            temp = temp->_next;
        }

        ListNode<T>* newNode = temp->_next;
        temp->_next = newNode->_next;
        delete newNode;
        _size--;
    }
}
