// reading a text file
#include "Graph.h"
#include "StackAndQueue.h"
#include "book.h"
#include "simpleLinkedListTemplate.h"
#include <fstream>
#include <iostream>
#include <string>

using namespace std;

///////////////////////////////////////////////////////////////////
// Submission Zone: You should submit your code after this line
///////////////////////////////////////////////////////////////////

template <class T> bool List<T>::exist(T n) {
  // You may leave this function (List<T>::exist(T n)) as it is, it's up to you
  // You *need to submit* this function, but your program should work without
  // modifying it I just put it here just in case that if anyone needs to
  // implement/modify this Meaning, you should be able to get full marks without
  // implementing this function
  return false;
}

template <class T> void List<T>::removeTail() {
  // You may leave this function (List<T>::removeTail()) as it is, it's up to
  // you You *need to submit* this function, but your program should work
  // without modifying it I just put it here just in case that if anyone needs
  // to implement/modify this Meaning, you should be able to get full marks
  // without implementing this function
}

// You should modify the following and submit

template <class T> void List<T>::insertTail(T n) {
  // Task 1
  if (_size == 0) {
    insertHead(n);
  } else {
    ListNode<T> *newNode = new ListNode<T>(n);
    _tail->_next = newNode;
    _tail = newNode;
    _size++;
  }
}

template <class T> void Stack<T>::push(T item) {
  // Task 2
  _l.insertHead(item);
}

template <class T> T Stack<T>::pop() {
  // Task 2
  T item = _l.headItem();
  _l.removeHead();
  return item;
}

template <class T> void Queue<T>::enqueue(T item) {
  // Task 3
  _l.insertTail(item);
}

template <class T> T Queue<T>::dequeue() {
  // Task 3
  T item = _l.headItem();
  _l.removeHead();
  return item;
}

int qBalancedBrackets(string exp) {
  // Task 4
  Stack<char> s;

  for (auto c : exp) {
    if (c == '(' || c == '{') {
      s.push(c);
    } else if (c == ')') {
      if (s.empty() || s.pop() != '(') {
        return 0;
      }
    } else if (c == '}') {
      if (s.empty() || s.pop() != '{') {
        return 0;
      }
    }
  }

  return s.empty() ? 1 : 0;
};

void Graph::DFS(int s, List<int> &output, bool resetVisited) {
  // Task 5
  if (resetVisited) {
    _resetVisited();
  }

  // iterative solution
  Stack<int> f;
  f.push(s);

  while (!f.empty()) {
    int c = f.pop();
    if (_isVisited(c))
      continue;
    _setVisited(c);
    output.insertTail(c);

    List<int> r;
    for (_al[c].start(); !_al[c].end(); _al[c].next()) {
      r.insertHead(_al[c].current());
    }

    for (r.start(); !r.end(); r.next()) {
      int n = r.current();
      if (!_isVisited(n)) {
        f.push(n);
      }
    }
  }

  // recursive solution
  // _setVisited(s);
  // output.insertTail(s);

  // for (_al[s].start(); !_al[s].end(); _al[s].next()) {
  //     int n = _al[s].current();
  //     if (!_isVisited(n)) {
  //         DFS(n, output, false);
  //     }
  // }
};

void Graph::BFS(int s, List<int> &output, bool resetVisited) {
  // Task 6
  if (resetVisited) {
    _resetVisited();
  }

  Queue<int> f;
  f.enqueue(s);

  while (!f.empty()) {
    int c = f.dequeue();
    if (_isVisited(c))
      continue;
    _setVisited(c);
    output.insertTail(c);

    for (_al[c].start(); !_al[c].end(); _al[c].next()) {
      int n = _al[c].current();
      if (!_isVisited(n)) {
        f.enqueue(n);
      }
    }
  }
};

int Graph::nComponents() {
  // Task 7
  _resetVisited();
  int n = 0;
  for (int s = 0; s < _nv; s++) {
    if (!_isVisited(s)) {
      n++;
      List<int> l;
      BFS(s, l, false);
    }
  }
  return n;
};

///////////////////////////////////////////////////////////////////
// Submission Zone Ended: You should submit your code above this line
///////////////////////////////////////////////////////////////////

void insertTailTest();
void stackTest();
void queueTest();
void bookStackTest();
void bookQueueTest();
int BFSTest(string, bool printGraph = true);
int DFSTest(string, bool printGraph = true);
int nComponentTest(string filename, bool printGraph = true);
void balancedBracketsTest(string exp);

int main() {
  // All test cases
  // Task 1
  insertTailTest();

  // Task 2
  stackTest();
  bookStackTest();
  // For the book test, you need to uncomment the function with body below

  // Task 3
  queueTest();
  bookQueueTest();

  // Task 4
  balancedBracketsTest("((a+b)+{e-(a-b)})");
  balancedBracketsTest("(((a+b)+{e-(a-b)})");
  balancedBracketsTest("(){}(){}(()){{}}");
  balancedBracketsTest(")(");

  cout << endl;

  // Task 5
  DFSTest("graphTraveralexample1.txt");
  DFSTest("graphTraveralexample2.txt");
  DFSTest("graphTraveralexample3.txt");

  // Task 6
  BFSTest("graphTraveralexample1.txt");
  BFSTest("graphTraveralexample2.txt");
  BFSTest("graphTraveralexample3.txt");

  // Task 7
  nComponentTest("cexample1.txt");
  nComponentTest("cexample2.txt");
  nComponentTest("cexample3.txt");

  return 0;
}

void insertTailTest() {
  cout << "Insert Tail Test:" << endl;
  List<int> li;
  for (int i = 1; i < 5; i++) {
    li.insertHead(i);
    li.insertTail(i);
  }
  cout << "List after inserting head and tails" << endl;
  cout << "Your list:" << endl;
  li.print();
  cout << "List after more deletion and insertion" << endl;
  for (int i = 0; i < 8; i++) {
    li.removeHead();
  }
  for (int i = 100; i < 104; i++) {
    li.insertHead(i);
    li.insertTail(i);
  }
  li.removeHead();
  cout << "Your list:" << endl;
  li.print();
  cout << endl;
}

void stackTest() {
  Stack<int> si;

  cout << "Stack Test:" << endl;

  for (int i = 0; i < 10; i++)
    si.push(i);
  for (int i = 0; i < 3; i++)
    si.pop();
  for (int i = 90; i < 100; i++)
    si.push(i);
  cout << "Your output" << endl;
  si.print();
  cout << endl;
}

void queueTest() {
  Queue<int> qi;
  cout << "Queue Test:" << endl;
  for (int i = 0; i < 10; i++)
    qi.enqueue(i);
  for (int i = 0; i < 3; i++)
    qi.dequeue();
  for (int i = 90; i < 100; i++)
    qi.enqueue(i);
  cout << "Your output" << endl;
  qi.print();
  cout << endl;
}

void bookStackTest() {
  Stack<Book> sb;

  cout << "Book Stack Test:" << endl;

  for (int i = 0; i < 5; i++) {
    string series("Hairy Border ");
    Book b(series + to_string(i + 1), 100 + i);
    sb.push(b);
  }

  for (int i = 0; i < 3; i++)
    sb.pop();
  for (int i = 0; i < 2; i++) {
    string series("Bravo Super Hero Comics ");
    Book b(series + to_string(i + 1), 23 + i);
    sb.push(b);
  }

  sb.print(true);
}

void bookQueueTest() {
  Queue<Book> qb;

  cout << "Book Queue Test:" << endl;

  for (int i = 0; i < 5; i++) {
    string series("Hairy Border ");
    Book b(series + to_string(i + 1), 100 + i);
    qb.enqueue(b);
  }

  for (int i = 0; i < 3; i++)
    qb.dequeue();
  for (int i = 0; i < 2; i++) {
    string series("Bravo Super Hero Comics ");
    Book b(series + to_string(i + 1), 23 + i);
    qb.enqueue(b);
  }

  qb.print(true);
}

void balancedBracketsTest(string exp) {
  cout << "The expression \"" << exp << "\" is "
       << (qBalancedBrackets(exp) ? "balanced" : "not balanced") << endl;
}

int DFSTest(string filename, bool printGraph) {
  cout << "DFS Test with file: " << filename << endl;
  ifstream myfile(filename);
  int v; // number of vertices
  int e; // number of edges
  int q; // number of queries

  if (myfile.fail()) {
    cout << "File not found" << endl;
    return -1;
  }

  myfile >> v;
  myfile >> e;
  myfile >> q;

  Graph G(v); // create a graph with nv nodes
  int s, d;

  // read in all the edges and add into the graph
  for (int i = 0; i < e; i++) {
    myfile >> s;
    myfile >> d;
    G.addEdge(s, d);
    G.addEdge(d, s);
  }

  if (printGraph)
    G.printGraph();

  for (int i = 0; i < q; i++) {
    myfile >> s;
    cout << "Starting from Node " << s << ":" << endl;
    List<int> li;
    G.DFS(s, li);
    li.print();
  }
  myfile.close();
  cout << endl;
  return 0;
}

int BFSTest(string filename, bool printGraph) {
  cout << "BFS Test with file: " << filename << endl;
  ifstream myfile(filename);
  int v; // number of vertices
  int e; // number of edges
  int q; // number of queries

  if (myfile.fail()) {
    cout << "File not found" << endl;
    return -1;
  }

  myfile >> v;
  myfile >> e;
  myfile >> q;

  Graph G(v); // create a graph with nv nodes
  int s, d;

  // read in all the edges and add into the graph
  for (int i = 0; i < e; i++) {
    myfile >> s;
    myfile >> d;
    G.addEdge(s, d);
    G.addEdge(d, s);
  }

  if (printGraph)
    G.printGraph();

  for (int i = 0; i < q; i++) {

    myfile >> s;
    cout << "Starting from Node " << s << ":" << endl;
    List<int> li;
    G.BFS(s, li);
    li.print();
  }
  myfile.close();
  cout << endl;
  return 0;
}

int nComponentTest(string filename, bool printGraph) {
  cout << "Number of Component Test with file: " << filename << endl;
  ifstream myfile(filename);
  int v; // number of vertices
  int e; // number of edges
  int q; // number of queries

  if (myfile.fail()) {
    cout << "File not found" << endl;
    return -1;
  }

  myfile >> v;
  myfile >> e;
  myfile >> q;

  Graph G(v); // create a graph with nv nodes
  int s, d;

  // read in all the edges and add into the graph
  for (int i = 0; i < e; i++) {
    myfile >> s;
    myfile >> d;
    G.addEdge(s, d);
    G.addEdge(d, s);
  }

  if (printGraph)
    G.printGraph();

  cout << "No. of components = " << G.nComponents() << endl;

  myfile.close();
  cout << endl;
  return 0;
}
