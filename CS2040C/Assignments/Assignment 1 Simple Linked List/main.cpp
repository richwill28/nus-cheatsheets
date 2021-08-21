#include <iostream>
#include "simpleIntLinkedList.h"
using namespace std;

int main()
{
	List l;
	l.insertHead(123);
	l.insertHead(11);
	l.insertHead(9);
	l.insertHead(1);
	l.insertHead(20);
	
	for (int i = 0; i < 5; i++) {
		cout << "The current list is: ";
		l.print();
		cout << "Does 9 exist in the list?" << (l.exist(9) ? "Yes" : "No") << endl << endl;
		l.removeHead();
	}
	return 0;
}
