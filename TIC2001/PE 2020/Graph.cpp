#include "Graph.h"
// uncomment this to include your own "heap.h"
// we will assume that you use the same code in your previous assignment
// #include "heap.h"

Graph::Graph(int n) {
  _al = new List<int>[n];
  _nv = n;
  _visited = new bool[n];
  _resetVisited();
}

void Graph::addEdge(int s, int d) { _al[s].insertHead(d); }

void Graph::printGraph() {
  for (int i = 0; i < _nv; i++) {
    cout << "Node " << i << ":";
    for (_al[i].start(); !_al[i].end(); _al[i].next())
      cout << " " << _al[i].current();
    cout << endl;
  }
}

Graph::~Graph() {
  delete[] _al;
  delete[] _visited;
}
