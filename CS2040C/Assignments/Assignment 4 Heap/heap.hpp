#pragma once
#include <math.h>
#include <iostream>
using namespace std;

#ifndef HEAPHPP
#define HEAPHPP

// iterative solution
template <class T>
void Heap<T>::_bubbleUp(int index) {
    while (index > 0) {
        int parent = floor((index - 1) / 2);
        if (_heap[index] < _heap[parent]) {
            break;
        }
        swap(_heap[index], _heap[parent]);
        index = parent;
    }
}

// recursive solution
// template <class T>
// void Heap<T>::_bubbleUp(int index) {
//     if (index <= 0) {
//         return;
//     }
//
//     int parent = floor((index - 1) / 2);
//     if (_heap[index] > _heap[parent]) {
//         swap(_heap[index], _heap[parent]);
//         _bubbleUp(parent);
//     }
// }

// iterative solution
template <class T>
void Heap<T>::_bubbleDown(int index) {
    while (index < _n) {
        int left = 2 * index + 1;
        int right = 2 * index + 2;
        if (left >= _n) {
            break;
        }

        int larger;
        if (left == _n - 1) {
            // implying right doesn't exist
            larger = left;
        } else {
            larger = _heap[left] > _heap[right] ? left : right;
        }

        if (_heap[index] > _heap[larger]) {
            break;
        }

        swap(_heap[index], _heap[larger]);
        index = larger;
    }
}

// recursive solution
// template <class T>
// void Heap<T>::_bubbleDown(int index) {
//     if (index >= _n) {
//         return;
//     }
//
//     int left = 2 * index + 1;
//     int right = 2 * index + 2;
//     if (left >= _n) {
//         return;
//     }
//
//     int larger;
//     if (left == _n - 1) {
//         // implying right doesn't exist
//         larger = left;
//     } else {
//         larger = _heap[left] > _heap[right] ? left : right;
//     }
//
//     if (_heap[index] < _heap[larger]) {
//         swap(_heap[index], _heap[larger]);
//         _bubbleDown(larger);
//     }
// }

template <class T>
void Heap<T>::insert(T item) {
    _heap[_n] = item;
    _bubbleUp(_n);
    _n++;
}

template <class T>
T Heap<T>::extractMax() {
    T item = _heap[0];
    deleteItem(_heap[0]);
    return item;
}

template <class T>
void Heap<T>::printHeapArray() {
    for (int i = 0; i < _n; i++) {
        cout << _heap[i] << " ";
    }
    cout << endl;
}

template <class T>
int Heap<T>::_lookFor(T x) {
    // not a very good implementation, but just use this for now.
    int i;
    for(i = 0; i < _n; i++) {
        if (_heap[i] == x) {
            return i;
        }
    }
    return -1;
}

template <class T>
void Heap<T>::decreaseKey(T from, T to) {
    int index = _lookFor(from);
    _heap[index] = to;
    _bubbleDown(index);
}

template <class T>
void Heap<T>::increaseKey(T from, T to) {
    int index = _lookFor(from);
    _heap[index] = to;
    _bubbleUp(index);
}

template <class T>
void Heap<T>::deleteItem(T x) {
    int index = _lookFor(x);
    swap(_heap[index], _heap[_n - 1]);
    _n--;
    _bubbleUp(index);
    _bubbleDown(index);
}

template <class T>
void Heap<T>::printTree() {
    int parity = 0;
    if (_n == 0) {
        return;
    }

    int space = pow(2, 1 + (int) log2f(_n)), i;
    int nLevel = (int) log2f(_n) + 1;
    int index = 0, endIndex;
    int tempIndex;
	
    for (int l = 0; l < nLevel; l++) {
        index = 1;
        parity = 0;
        for (i = 0; i < l; i++) {
            index *= 2;
        }

        endIndex = index * 2 - 1;
        index--;
        tempIndex = index;
        while (index < _n && index < endIndex) {
            for (i = 0; i < space-1; i++) {
                cout << " ";
            }

            if (index == 0) {
                cout << "|";
            } else if (parity) {
                cout << "\\";
            } else {
                cout << "/";
            }

            parity = !parity;
            for (i = 0; i < space; i++) {
                cout << " ";
            }

            index++;
        }

        cout << endl;
        index = tempIndex;
        while (index < _n && index < endIndex) {
            for (i = 0; i < (space - 1 - ((int) log10(_heap[index]))); i++) {
                cout << " ";
            }

            cout << _heap[index];
            for (i = 0; i < space; i++) {
                cout << " ";
            }

            index++;
        }

        cout << endl;
        space /= 2;
    }
}

#endif
