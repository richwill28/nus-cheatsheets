#pragma once

#include <iostream>
#include <string>

using namespace std;

class Book {
private:
  string _name;
  int _nPages;

public:
  Book() {
    _name = "";
    _nPages = 0;
  };

  Book(string name, int n) {
    _name = name;
    _nPages = n;
  };

  bool operator==(const Book &b) {
    return (_name == b._name) && (_nPages == b._nPages);
  };

  string name() { return _name; };
  int nPages() { return _nPages; };

  friend ostream &operator<<(ostream &os, const Book &f) {
    os << f._name << " with " << f._nPages << " pages" << endl;
    ;
    return os;
  }
};
