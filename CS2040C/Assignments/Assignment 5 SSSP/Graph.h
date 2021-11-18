#pragma once
#include "simpleLinkedListTemplate.h"

class nodeWeightPair {
    private:
        int _node;
        int _weight;

    public:
        nodeWeightPair() { _node = -1; _weight = -1; } // a constructor that shouldn't be used.
        nodeWeightPair(int n, int w) { _node = n; _weight = w; }
        nodeWeightPair(const nodeWeightPair& nwp) { _node = nwp._node; _weight = nwp._weight; }
        int nodeIndex() { return _node; }
        int weight() { return _weight; }
        bool operator>(const nodeWeightPair& nwp) { return _weight > nwp._weight; }
        bool operator<(const nodeWeightPair& nwp) { return _weight < nwp._weight; }
        bool operator==(const nodeWeightPair& nwp) { return _node == nwp._node; }
        friend ostream& operator<<(ostream&, const nodeWeightPair&);
};

class Graph {
    private:
        List<nodeWeightPair> *_al; // adjaceny list
        int _nv; // number of nodes

    public:
        Graph(int n);
        void addEdge(int s, int d, int w);
        int shortestDistance(int s, int d);
        void printGraph();
        ~Graph();
};
