#include <iostream>

template <class T>
class LinkedList;

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
    
    friend class LinkedList<T>;
};

template <class T>
class LinkedList {
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
            newNode->next = head;
            head = newNode;
            size++;
        }

        // Time complexity: O(n)
        void insertTail(T item) {
            if (size == 0) {
                insertHead(item);
            } else {
                Node<T> *newNode = new Node<T>(item);
                Node<T> *tempNode = head;
                while (tempNode->next != NULL) {
                    tempNode = tempNode->next;
                }
                tempNode->next = newNode;
                size++;
            }
        }

        // Time complexity: O(n)
        void insertAtIndex(T item, int index) {
            if (index < 0 || index > size) {
                return;
            }

            if (index == 0) {
                insertHead(item);
            } else {
                Node<T> *newNode = new Node<T>(item);
                Node<T> *tempNode = head;
                for (int i = 1; i < index; i++) {
                    tempNode = tempNode->next;
                }
                newNode->next = tempNode->next;
                tempNode->next = newNode;
                size++;
            }
        }

        // Time complexity: O(1)
        void removeHead() {
            if (size == 0) {
                return;
            }

            Node<T> *tempNode = head;
            head = head->next;
            tempNode->next = NULL;
            delete tempNode;
            size--;
        }

        // Time complexity: O(n)
        void removeTail() {
            if (size == 0) {
                return;
            }

            Node<T> *prevNode = NULL;
            Node<T> *currNode = head;
            while (currNode->next != NULL) {
                prevNode = currNode;
                currNode = currNode->next;
            }
            prevNode->next = NULL;
            delete currNode;
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
                Node<T> *newNode = new Node<T>(item);
                Node<T> *prevNode = NULL;
                Node<T> *currNode = head;
                for (int i = 0; i < index; i++) {
                    prevNode = currNode;
                    currNode = currNode->next;
                }
                prevNode->next = currNode->next;
                currNode->next = NULL;
                delete currNode;
                size--;
            }
        }

        // Time complexity: O(1)
        T getHeadItem() {
            if (size == 0) {
                return -1;
            }
            return head->item;
        }

        // Time complexity: O(n)
        T getTailItem() {
            if (size == 0) {
                return -1;
            }

            Node<T> *tempNode = head;
            while (tempNode->next != NULL) {
                tempNode = tempNode->next;
            }
            return tempNode->item;
        }

        // Time complexity: O(n)
        T getItemAtIndex(int index) {
            if (size == 0 || index >= size) {
                return -1;
            }

            Node<T> *tempNode = head;
            for (int i = 0; i < index; i++) {
                tempNode = tempNode->next;
            }
            return tempNode->item;
        }

        // Time complexity: O(n)
        bool exist(T item) {
            Node<T> *tempNode = head;
            while (tempNode != NULL) {
                if (tempNode->item == item) {
                    return true;
                }
                tempNode = tempNode->next;
            }
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
            Node<T> *prevNode = NULL;
            Node<T> *currNode = head;
            Node<T> *nextNode = NULL;
            while (currNode != NULL) {
                nextNode = currNode->next;
                currNode->next = prevNode;
                prevNode = currNode;
                currNode = nextNode;
            }
            head = prevNode;
        }

        // Time complexity: O(n)
        ~LinkedList() {
            while (size > 0) {
                removeHead();
            }
        }

        // to be replaced with better implementation
        void print() {
            Node<T> *tempNode = head;
            while (tempNode != NULL) {
                std::cout << tempNode->item << " ";
                tempNode = tempNode->next;
            }
        }
};
