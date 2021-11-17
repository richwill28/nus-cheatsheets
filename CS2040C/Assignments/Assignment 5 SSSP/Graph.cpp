#include "Graph.h"
#include "heap.h"

std::ostream& operator<<(std::ostream& os, nodeWeightPair const& n) {
    return os << " (idx:" << n._node << " w:" << n._weight << ")";
}

Graph::Graph(int n) {
    _al = new List<nodeWeightPair>[n];
    _nv = n;
}

int Graph::shortestDistance(int s, int d) {
    // setup
    Heap<nodeWeightPair> PQ;
    PQ.insert(nodeWeightPair(s, 0));

    int dist[_nv];
    bool visited[_nv];
    int parent[_nv];

    for (int i = 0; i < _nv; i++) {
        dist[i] = 2147483647;   // initialize distance to INT_MAX
        visited[i] = false;
        parent[i] = -1;
    }

    dist[s] = 0;
    visited[s] = true;

    // main routine of Dijkstra
    while (!PQ.empty()) {
        nodeWeightPair node = PQ.extractMax();
        int i = node.nodeIndex();
        // traverse neighboring nodes
        for (_al[i].start(); !_al[i].end(); _al[i].next()) {
            int dest = _al[i].current().nodeIndex();
            int weight = _al[i].current().weight();
            // relaxation
            if (!visited[dest] && (dist[i] + weight < dist[dest])) {
                if (dist[dest] != 2147483647) {
                    // delete old node if already exist in PQ
                    PQ.deleteItem(nodeWeightPair(dest, (-1) * dist[dest]));
                }
                dist[dest] = dist[i] + weight;  // update distance
                parent[dest] = i;   // update parent
                PQ.insert(nodeWeightPair(dest, (-1) * dist[dest])); // update node in PQ
            }
        }
        visited[i] = true;  // prevent visiting extracted node
    }

    if (dist[d] == 2147483647) {
        // if distance is never updated
        return -1;
    }

    // main routine for printing path
    List<int> path;
    path.insertHead(d);
    int node = parent[d];
    while (node >= 0) {
        path.insertHead(node);
        node = parent[node];
    }
    cout << "Path:";
    for (path.start(); !path.end(); path.next()) {
        cout << " " << path.current();
    }
    cout << endl;

    return dist[d];
}

void Graph::addEdge(int s, int d, int w) {
    _al[s].insertHead(nodeWeightPair(d, w));
}

void Graph::printGraph() {
    for (int i=0; i < _nv; i++) {
        cout << "Node " << i << ": ";
        for (_al[i].start(); !_al[i].end(); _al[i].next()) {
            cout << " (" << _al[i].current().nodeIndex() << "," << _al[i].current().weight() << ")";
        }
        cout << endl;
    }
}

Graph::~Graph() {
    delete[] _al;
}
