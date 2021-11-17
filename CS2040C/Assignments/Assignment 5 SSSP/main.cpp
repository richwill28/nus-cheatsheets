// compile code with g++ main.cpp Graph.cpp
#include <iostream>
#include <fstream>
#include <string>
#include "Graph.h"
using namespace std;

int main() {
    string line;

    // ifstream myfile("example1.txt");
    // ifstream myfile("example2.txt");
    // ifstream myfile("example3.txt");
    // ifstream myfile("example4.txt");
    // ifstream myfile("example5.txt");
    ifstream myfile("train.txt");

    int v;  // number of vertices
    int e;  // number of edges
    int q;  // number of queries
    if (myfile.fail()) {
        cout << "File not found" << endl;
        return -1;
    }

    myfile >> v;
    myfile >> e;
    myfile >> q;

    Graph G(v); // create a graph with nv nodes
    int s, d, w;

    // read in all the edges and add into the graph
    for (int i = 0; i < e; i++) {
        myfile >> s;
        myfile >> d;
        myfile >> w;
        G.addEdge(s, d, w);
    }

    G.printGraph();

    for (int i = 0; i < q; i++) {
        myfile >> s;
        myfile >> d;
        int dist = G.shortestDistance(s, d);
        if (dist == -1) {
            cout << "Node " << s << " and Node " << d << " are not connected in the same component." << endl;
        } else {
            cout << "The shortest distance from the vertex " << s << " to vertex " << d << " is " << dist << endl;
        }
    }

    myfile.close();
    return 0;
}
