void HashTable::insert(int n) {
    int idx = h(n) % _size;
    int numOfCollision = 0;
    while (_ht[idx] != 0 && _ht[idx] != -1) {
        if (_ht[idx] == n) {
            cout << n << " already exists in the hash table." << endl;
            return;
        }
        numOfCollision++;
        idx = (h(n) + numOfCollision * numOfCollision) % _size;
    }

    _ht[idx] = n;
    _nItem++;
}

void HashTable::remove(int n) {
    if (!exist(n)) {
        cout << "Fail to remove " << n << endl;
        return;
    }

    int idx = h(n) % _size;
    int numOfCollision = 0;
    while (_ht[idx] != n) {
        numOfCollision++;
        idx = (h(n) + numOfCollision * numOfCollision) % _size;
    }

    _ht[idx] = -1;
    _nItem--;
}

bool HashTable::exist(int n) {
    int idx = h(n) % _size;
    int numOfCollision = 0;
    while (_ht[idx] != 0) {
        if (_ht[idx] == n) {
            return true;
        }
        numOfCollision++;
        idx = (h(n) + numOfCollision * numOfCollision) % _size;
    }

    return false;
}

void HashTable::resize(int newSize) {
    int* temp = _ht;
    _ht = new int[newSize];
    for (int i = 0; i < newSize; i++) {
        _ht[i] = 0;
    }

    int oldsize = _size;
    _size = newSize;
    _nItem = 0;
    for (int i = 0; i < oldsize; i++) {
        if (temp[i] != 0 && temp[i] != -1) {
            insert(temp[i]);
        }
    }

    delete[] temp;
}

// O(n^2) implementation
int n3Sum(int* arr, int size, int total) {
    HashTable ht(size * 10);    // any arbitrary factor is fine as long as size is large enough
    for (int i = 0; i < size; i++) {
        ht.insert(arr[i]);
    }

    int count = 0;
    for (int i = 0; i < size; i++) {
        for (int j = i + 1; j < size; j++) {
            int diff = total - arr[i] - arr[j];
            if (diff != arr[i] && diff != arr[j] && ht.exist(diff)) {
                count++;
            }
        }
    }

    return count / 3;
}

// O(n^3) implementation
// int n3Sum(int* arr, int size, int total) {
//     int count = 0;
//     for (int i = 0; i < size; i++) {
//         for (int j = i + 1; j < size; j++) {
//             for (int k = j + 1; k < size; k++) {
//                 if (arr[i] + arr[j] + arr[k] == total) {
//                     count++;
//                 }
//             }
//         }
//     }
//     return count;
// }
