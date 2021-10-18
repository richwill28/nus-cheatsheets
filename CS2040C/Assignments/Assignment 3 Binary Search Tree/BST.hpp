#pragma once
#ifndef BSTHPP
#define BSTHPP
#include "BST.h"

template <class T>
void BinarySearchTree<T>::insert(T x) {
    if (exist(x)) {
        return;
    }

    if (_root == NULL) {
        _root = new TreeNode<T>(x);
    } else {
        _root = _insert(_root, x);
    }
    _size++;
}

template <class T>
TreeNode<T>* BinarySearchTree<T>::_insert(TreeNode<T>* current, T x) {
    if (x < current->_item) {
        if (current->_left) {
            current->_left = _insert(current->_left, x);
        } else {
            current->_left = new TreeNode<T>(x);
        }
    } else if (x > current->_item) {
        if (current->_right) {
            current->_right = _insert(current->_right, x);
        } else {
            current->_right = new TreeNode<T>(x);
        }
    }

    current->_height = _updateHeight(current);
    current = _balanceNode(current);
    return current;
}

template <class T>
bool BinarySearchTree<T>::exist(T x) {
    TreeNode<T>* currentNode = _root;
    while (currentNode) {
        if (x < currentNode->_item) {
            currentNode = currentNode->_left;
        } else if (x > currentNode->_item) {
            currentNode = currentNode->_right;
        } else {
            return true;
        }
    }
    return false;
}

template <class T>
T BinarySearchTree<T>::searchMax() {
    return _searchMax(_root);
}

template <class T>
T BinarySearchTree<T>::_searchMax(TreeNode<T>* node) {
    TreeNode<T>* currentNode = node;
    while (currentNode->_right) {
        currentNode = currentNode->_right;
    }
    return currentNode->_item;
}

template <class T>
T BinarySearchTree<T>::searchMin() {
    return _searchMin(_root);
}

template <class T>
T BinarySearchTree<T>::_searchMin(TreeNode<T>* node) {
    TreeNode<T>* currentNode = node;
    while (currentNode->_left) {
        currentNode = currentNode->_left;
    }
    return currentNode->_item;
}

template <class T>
T BinarySearchTree<T>::successor(T x) {
    TreeNode<T>* successorNode = NULL;
    TreeNode<T>* currentNode = _root;
    while (currentNode) {
        if (x < currentNode->_item) {
            successorNode = currentNode;
            currentNode = currentNode->_left;
        } else {
            currentNode = currentNode->_right;
        }
    }
    return successorNode->_item;
}

template <class T>
int BinarySearchTree<T>::_getHeight(TreeNode<T>* node) {
    return node ? node->_height : -1;
}

template <class T>
int BinarySearchTree<T>::_balanceFactor(TreeNode<T>* node) {
    return _getHeight(node->_right) - _getHeight(node->_left);
}

template <class T>
int BinarySearchTree<T>::_updateHeight(TreeNode<T>* node) {
    int leftHeight = _getHeight(node->_left);
    int rightHeight = _getHeight(node->_right);
    return max(leftHeight, rightHeight) + 1;
}

template <class T>
TreeNode<T>* BinarySearchTree<T>::_balanceNode(TreeNode<T>* node) {
    if (_balanceFactor(node) < -1) {
        // left heavy

        TreeNode<T>* child = node->_left;
        if (_balanceFactor(child) >= 1) {
            // left child is right heavy
            child = _leftRotation(child);
        }
        return _rightRotation(node);
    } else if (_balanceFactor(node) > 1) {
        // right heavy

        TreeNode<T>* child = node->_right;
        if (_balanceFactor(child) <= -1) {
            // right child is left heavy
            child = _rightRotation(child);
        }
        return _leftRotation(node);
    }

    return node;
}

template <class T>
TreeNode<T>* BinarySearchTree<T>::_leftRotation(TreeNode<T>* node) {
    // main routine
    TreeNode<T>* parent = node->_right;
    node->_right = parent->_left;
    parent->_left = node;

    // update heights
    node->_height = _updateHeight(node);
    parent->_height = _updateHeight(parent);

    return parent;
}

template <class T>
TreeNode<T>* BinarySearchTree<T>::_rightRotation(TreeNode<T>* node) {
    // main routine
    TreeNode<T>* parent = node->_left;
    node->_left = parent->_right;
    parent->_right = node;

    // update heights
    node->_height = _updateHeight(node);
    parent->_height = _updateHeight(parent);

    return parent;
}

template <class T>
void BinarySearchTree<T>::printTree(bool withHeight) {
    _printTree(0, _root, withHeight);
}

template <class T>
void BinarySearchTree<T>::preOrderPrint() {
    _preOrderPrint(_root);
    cout << endl;
}

template <class T>
void BinarySearchTree<T>::_preOrderPrint(TreeNode<T>* node) {
    if (!node) {
        return;
    }

    cout << node->_item << " ";
    _preOrderPrint(node->_left);
    _preOrderPrint(node->_right);
}

template <class T>
void BinarySearchTree<T>::inOrderPrint() {
    _inOrderPrint(_root);
    cout << endl;
}

template <class T>
void BinarySearchTree<T>::_inOrderPrint(TreeNode<T>* node) {
    if (!node) {
        return;
    }

    _inOrderPrint(node->_left);
    cout << node->_item << " ";
    _inOrderPrint(node->_right);
}

template <class T>
void BinarySearchTree<T>::postOrderPrint() {
    _postOrderPrint(_root);
    cout << endl;
}

template <class T>
void BinarySearchTree<T>::_postOrderPrint(TreeNode<T>* node) {
    if (!node) {
        return;
    }

    _postOrderPrint(node->_left);
    _postOrderPrint(node->_right);
    cout << node->_item << " ";
}

template <class T>
void BinarySearchTree<T>::_printTree(int indent, TreeNode<T>* node, bool withHeight) {
    if (!node) {
        return;
    }

    if (node->_right) {
        _printTree(indent + 2, node->_right, withHeight);
    }

    for (int i = 0; i < indent; i++) {
        cout << "  ";
    }
    cout << node->_item;

    if (withHeight) {
        cout << "(h=" << node->_height << ")";
    }
    cout << endl;

    if (node->_left) {
        _printTree(indent + 2, node->_left, withHeight);
    }
};

template <class T>
void BinarySearchTree<T>::_destroySubTree(TreeNode<T>* node) {
    if (node->_left) {
        _destroySubTree(node->_left);
    }

    if (node->_right) {
        _destroySubTree(node->_right);
    }

    delete node;
}

template <class T>
BinarySearchTree<T>::~BinarySearchTree() {
    if (_root) {
        _destroySubTree(_root);
    }
}

#endif
