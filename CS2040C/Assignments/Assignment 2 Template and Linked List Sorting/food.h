#pragma once
#include <string>
using namespace std;

class Food {
    private:
        string _name;
        int _cal;

	public:
        Food() { _name = ""; _cal = 0; };
        Food(string, int);
        Food operator+(const Food&);
        bool operator>(const Food&);
        bool operator==(const Food&);
        string name() { return _name; };
        int cal() { return _cal; };
        friend ostream &operator<<(ostream&, const Food&);
};

#include "food.cpp"
