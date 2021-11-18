#pragma once
#include <iostream>
using namespace std;

class HashTable {
    private:
        int* _ht;       // array to store hash table items
        int _size;      // size of the hash table
        int _nItem;     // number of items added
        int h(int x);   // hash function

    public:
        HashTable(int);
        void insert(int);
        void remove(int);
        bool exist(int);
        void resize(int newSize);
        void printHashTable();
        ~HashTable();
};

#include "HashTable.hpp"
