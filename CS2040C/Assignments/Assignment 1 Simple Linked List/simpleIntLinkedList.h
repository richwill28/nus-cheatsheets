#pragma once

class ListNode {
    private:
        int _item;
        ListNode *_next;

    public:
        ListNode(int);

        int content() {
            return _item;
        }

    friend class List;
};

class List {
    private:
        int _size;
        ListNode *_head;

    public:
        List() {
            _size = 0;
            _head = NULL;
        }

        ~List();
        void insertHead(int);
        void removeHead();
        void print();
        bool exist(int);
        int headItem();
        bool empty();
        int tailItem();
        void removeTail();
};
