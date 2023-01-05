#pragma once
#include "simpleLinkedListTemplate.h"

class Graph {
private:
  List<int> *_al; // adjaceny list
  int _nv;        // number of nodes
  bool *_visited;
  int _nVisited;

  void _resetVisited() {
    for (int i = 0; i < _nv; i++)
      _visited[i] = false;
    _nVisited = 0;
  };

  void _setVisited(int node) {
    if (_visited[node] == false)
      _nVisited++;
    _visited[node] = true;
  };

  bool _isVisited(int node) { return _visited[node]; };

public:
  Graph(int n);
  void addEdge(int s, int d);
  void BFS(int s, List<int> &output, bool resetVisited = true);
  void DFS(int s, List<int> &output, bool resetVisited = true);
  int nComponents();
  void printGraph();
  ~Graph();
};
