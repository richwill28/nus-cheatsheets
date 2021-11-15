#include <iostream>

template <class T>
class CircularLinkedList;

template <class T>
class Node {
    private:
        T item;
        Node<T> *next;

    public:
        Node(T item) {
            this->item = item;
            this->next = NULL;
        }
    
    friend class CircularLinkedList<T>;
};

template <class T>
class CircularLinkedList {
    private:
        int size;
        Node<T> *head;

    public:
        // Time complexity: O(1)
        LinkedList() {
            this->size = 0;
            this->head = NULL;
        }

        // Time complexity: O(1)
        void insertHead(T item) {
            Node<T> *newNode = new Node<T>(item);
            if (size == 0) {
                head = newNode;
                newNode->next = newNode;
            } else {
                newNode->item = head->item;
                newNode->next = head->next;
                head->item = item;
                head->next = newNode;
            }
            size++;
        }

        // Time complexity: O(n)
        void insertTail(T item) {
            if (size == 0) {
                insertHead(item);
            } else {
                Node<T> *newNode = new Node<T>(item);
                Node<T> *tempNode = head;
                while (tempNode->next != head) {
                    tempNode = tempNode->next;
                }
                tempNode->next = newNode;
                newNode->next = head;
                size++;
            }
        }

        // Time complexity: O(n)
        void insertAtIndex(T item, int index) {
            if (index < 0 || index > size) {
                return;
            }

            if (index == 0) {
                insertAtIndex(item);
            } else {
                Node<T> *newNode = new Node<T>(item);
                Node<T> *prevNode = head;
                Node<T> *nextNode = head->next;
                for (int i = 0; i < index - 1; i++) {
                    prevNode = nextNode;
                    nextNode = nextNode->next;
                }
                prevNode->next = newNode;
                newNode->next = nextNode;
                size++;
            }
        }

        // Time complexity: O(1)
        void removeHead() {
            if (size == 0) {
                return;
            }

            head->item = head->next->item;
            Node<T> *tempNode = head->next;
            head->next = tempNode->next;
            tempNode->next = NULL;
            delete tempNode;
            size--;
        }

        // Time complexity: O(n)
        void removeTail() {
            if (size == 0) {
                return;
            }

            Node<T> *prevNode = head;
            Node<T> *tempNode = head->next;
            while (tempNode->next != head) {
                prevNode = tempNode;
                tempNode = tempNode->next;
            }
            prevNode->next = tempNode->next;
            tempNode->next = NULL;
            delete tempNode;
            size--;
        }

        // Time complexity: O(n)
        void removeAtIndex(T item, int index) {
            if (index < 0 || index >= size) {
                return;
            }

            if (index == 0) {
                removeHead(item);
            } else {
                Node<T> *prevNode = head;
                Node<T> *tempNode = head->next;
                for (int i = 0; i < index - 1; i++) {
                    prevNode = tempNode;
                    tempNode = tempNode->next;
                }
                prevNode->next = tempNode->next;
                tempNode->next = NULL;
                delete tempNode;
                size--;
            }
        }

        // Time complexity: O(1)
        T getHeadItem() {
            if (size == 0) {
                return -1;    // to be replaced with better implementation
            }
            return head->item;
        }

        // Time complexity: O(n)
        T getTailItem() {
            if (size == 0) {
                return -1;    // to be replaced with better implementation
            }

            Node<T> *tempNode = head;
            while (tempNode->next != head) {
                tempNode = tempNode->next;
            }
            return tempNode->item;
        }

        // Time complexity: O(n)
        T getItemAtIndex(int index) {
            if (size == 0 || index >= size) {
                return -1;    // to be replaced with better implementation
            }

            Node<T> *tempNode = head;
            for (int i = 0; i < index; i++) {
                tempNode = tempNode->next;
            }
            return tempNode->item;
        }

        // Time complexity: O(n)
        bool exist(T item) {
            if (size == 0) {
                return false;
            }

            Node<T> *tempNode = head;
            do {
                if (tempNode->item == item) {
                    return true;
                }
                tempNode = tempNode->next;
            } while (tempNode->next != head);
            return false;
        }

        // Time complexity: O(1)
        bool isEmpty() {
            return (size == 0);
        }

        // Time complexity: O(n)
        void rotateList(int offset) {
            if (size == 0) {
                return;
            }

            offset = offset % size;
            for (int i = 0; i < offset; i++) {
                head = head->next;
            }
        }

        // Time complexity: O(n)
        void reverseList() {
            if (size <= 1) {
                return;
            }

            Node<T> *prevNode = NULL;
            Node<T> *currNode = head;
            Node<T> *nextNode = NULL;
            do {
                nextNode = currNode->next;
                currNode->next = prevNode;
                prevNode = currNode;
                currNode = nextNode;
            } while (currNode != head);
            head->next = prevNode;
            head = prev;
        }

        // Time complexity: O(n)
        ~LinkedList() {
            while (size > 0) {
                removeHead();
            }
        }
};
