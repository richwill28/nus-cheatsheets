#include <iostream>
#include "simpleLinkedListTemplate.h"
#include "food.h"
using namespace std;

void testIntLL();
void testFoodOpGreaterThan();
void testFoodSort();
void testReverseOp();
void testFoodExist();
void testFoodAddition();
void testIntLLExtractMax();

int main() {
	testIntLL();
	testFoodExist();
	testFoodOpGreaterThan();
	testFoodAddition();
	testIntLLExtractMax();
	testFoodSort();
	testReverseOp();
	return 0;
}

void testIntLL() {
	cout << endl << "Testing List<int>" << endl;
	List<int> l;

	// testing code for LinkedList<int>
	l.insertHead(123);
	l.insertHead(11);
	l.insertHead(9);
	l.insertHead(1);
	l.insertHead(20);

	cout << "This is the linked list we have:" << endl;
	l.print(false);

	cout << endl << "Testing the function exist()" << endl;
	int testArr[] = { 10,20,30,40,50 };
	for (int i = 0; i < 5; i++) {
		cout << "The number " << testArr[i] << " is " << (l.exist(testArr[i]) ? "" : "not ") << "in the array" << endl;
	}

	// code for testing extractHead() 
	cout << endl << "Here is the sorted list of number in decending order" << endl;
	while (!l.empty()) {
		cout << l.extractMax() << " " ;
	}
	cout << endl;
}

void testFoodOpGreaterThan() {
	cout << endl << "Testing operator \">\" for class Food" << endl;
	List<Food> l_food;

	// code for testing the operator ">" on Food
	Food food1("Salad", 100);
	Food food2("French Fries", 10000);

	cout << "Among " << food1.name() << " and " << food2.name() << "..." << endl;
	cout << "The food with more calories is ";
	cout << (food1 > food2 ? food1 : food2) << endl;
}

void testFoodExist() {
	cout << endl << "Testing exist for List<Food>" << endl;
	List<Food> l_food;

	// code for LinkedList<Food>
	l_food.insertHead(Food("Beef", 300));
	l_food.insertHead(Food("Rice", 500));
	l_food.insertHead(Food("Chocolate", 200));
	l_food.insertHead(Food("Pork Chop", 150));
	l_food.insertHead(Food("Chicken Chop", 100));
	l_food.insertHead(Food("Salad", 50));
	l_food.insertHead(Food("Fish", 100));
	l_food.insertHead(Food("Veggies", 100));
	l_food.insertHead(Food("Soup", 50));

	cout << "The food \"Fish\" " << (!l_food.exist(Food("Fish", 0)) ? "does not exist " : "exists ") << "in the list" << endl;
	cout << "The food \"Banana\" " << (!l_food.exist(Food("Banana", 0)) ? "does not exist " : "exists ") << "in the list" << endl;
	cout << "The food \"Fish Soup\" " << (!l_food.exist(Food("Fish Soup", 0)) ? "does not exist " : "exists ") << "in the list" << endl;
}

void testIntLLExtractMax() {
	cout << endl << "Testing Extract Maximum for List<int>" << endl;
	List<int> l;

	// testing code for LinkedList<int>
	l.insertHead(123);
	l.insertHead(11);
	l.insertHead(9);
	l.insertHead(1);
	l.insertHead(20);

	cout << "This is the linked list we have:" << endl;
	l.print(false);

	cout << endl << "After one extractMax()" << endl;
	l.extractMax();
	l.print(false);

	cout << endl << "After another one extractMax()" << endl;
	l.extractMax();
	l.print(false);
}

void testFoodSort() {
	cout << endl << "Testing extractMax() for class Food" << endl;
	List<Food> l_food;

	// code for LinkedList<Food>
	l_food.insertHead(Food("Beef", 300));
	l_food.insertHead(Food("Rice", 500));
	l_food.insertHead(Food("Chocolate", 200));
	l_food.insertHead(Food("Pork Chop", 150));
	l_food.insertHead(Food("Chicken Chop", 100));
	l_food.insertHead(Food("Salad", 50));
	l_food.insertHead(Food("Fish", 100));
	l_food.insertHead(Food("Veggies", 100));
	l_food.insertHead(Food("Soup", 50));

	cout << "The food Fish " << (!l_food.exist(Food("Fish",0)) ? "does not exist " : "exists ") << "in the list" << endl;
	cout << "The food Banana " << (!l_food.exist(Food("Banana",0)) ? "does not exist " : "exists ") << "in the list" << endl;
	cout << "The food Fish Soup" << (!l_food.exist(Food("Fish Soup",0)) ? "does not exist " : "exists ") << "in the list" << endl;

	cout << endl << "Here is the list of food stored, according to the list order from head to tail:" << endl;
	l_food.print(true);

	cout << endl << "The sorted list of food in decending order is: " << endl;
	while (!l_food.empty()) {
		cout << l_food.extractMax() << endl;
	}
	cout << endl;
} 

void testReverseOp() {
	cout << endl << "Testing reverseOp()" << endl;
	List<int> l;

	// testing code for LinkedList<int>
	l.insertHead(123);
	l.insertHead(11);
	l.insertHead(9);
	l.insertHead(1);
	l.insertHead(20);

	cout << "This is the linked list we have:" << endl;
	l.print(false);

	l.reverseOp();
	cout << "This is the linked list after reverseOp():" << endl;
	l.print(false);

	l.reverseOp();
	cout << "This is the linked list after reverseOp() again:" << endl;
	l.print(false);

	l.reverseOp();
	cout << "This is the linked list after reverseOp() again and again" << endl;
	l.print(false);

	l.reverseOp();
	cout << "This is the linked list after reverseOp() again and again and again" << endl;
	l.print(false);
}

void testFoodAddition() {
	cout << endl << "Testing operator \"+\" for class Food" << endl;
	List<Food> l_food;

	// code for testing the operator ">" on Food
	Food food1("Salad", 100);
	Food food2("Chicken", 200);
	Food food3("Curry", 40);
	Food food4("Ice Cream", 300);

	Food food23 = food2 + food3;
	Food food21 = food2 + food1;
	Food food31 = food3 + food1;
	Food foodALL = food3 + food2 + food4 + food4;

	l_food.insertHead(food1);
	l_food.insertHead(food2);
	l_food.insertHead(food3);
	l_food.insertHead(food4);
	l_food.insertHead(food23);
	l_food.insertHead(food21);
	l_food.insertHead(food31);
	l_food.insertHead(foodALL);

	l_food.print(true);
}
