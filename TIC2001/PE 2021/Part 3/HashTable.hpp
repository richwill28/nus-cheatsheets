#pragma once
#include "HashTable.h"

int HashTable::h(int x) {
    // sum of all digits
    int ans = 0;
    while (x > 0) {
        int lastDigit = x % 10;
        ans += lastDigit;
        x /= 10;
    }
    return ans;
}

HashTable::HashTable(int n) {
    // initially, populate hash table with 0
    _size = n;
    _nItem = 0;
    _ht = new int[_size];
    for (int i = 0; i < _size; i++) {
        _ht[i] = 0;
    }
}

void HashTable::printHashTable() {	
    cout << "Current hash table: " << endl;
    for (int i = 0; i < _size; i++) {
        cout << _ht[i] << " ";
    }
    cout << endl;
}

HashTable::~HashTable() {
    delete[] _ht;
}
