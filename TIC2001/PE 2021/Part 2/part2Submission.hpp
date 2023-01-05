template <class T> void DHeap<T>::_bubbleUp(int index) {
  while (index) {
    int parent = (index - 1) / _d;
    if (_heap[parent] > _heap[index]) {
      break;
    }
    swap(_heap[index], _heap[parent]);
    index = parent;
  }
}

template <class T> void DHeap<T>::_bubbleDown(int index) {
  while (index < _n) {
    int indexOfLargestChild = _d * index + 1;
    if (indexOfLargestChild > _n) {
      break;
    }

    for (int i = 2; i <= _d; i++) {
      int currentIndex = _d * index + i;
      bool isIndexInRange = currentIndex < _n;
      bool isLarger = _heap[currentIndex] > _heap[indexOfLargestChild];

      if (isIndexInRange && isLarger) {
        indexOfLargestChild = currentIndex;
      }
    }

    if (_heap[index] > _heap[indexOfLargestChild]) {
      break;
    }

    swap(_heap[index], _heap[indexOfLargestChild]);
    index = indexOfLargestChild;
  }
}

template <class T> T DHeap<T>::extractMax() {
  // assuming _n > 0
  T max = _heap[0];
  _heap[0] = _heap[_n - 1];
  _n--;
  _bubbleDown(0);
  return max;
}

template <class T> void DHeap<T>::insert(T item) {
  _heap[_n] = item;
  _bubbleUp(_n);
  _n++;
}
