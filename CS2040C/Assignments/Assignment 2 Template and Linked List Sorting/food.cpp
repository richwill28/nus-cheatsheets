#include "food.h"
#include <iostream>
using namespace std;

Food::Food(string s, int cal) {
	_name = s;
	_cal = cal;
}

Food Food:: operator+(const Food& f) {
	return Food(_name + " " + f._name, _cal + f._cal);
}


bool Food:: operator>(const Food& f) {
	return _cal > f._cal;
}

bool Food:: operator==(const Food& f) {
	return true;
}

ostream& operator<<(ostream& os, const Food& f) {
	os << f._name << " with " << f._cal << " calories" ;
	return os;
}
