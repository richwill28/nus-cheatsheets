#include "HashTable.h"
#include "part3Submission.hpp"
#include <time.h>

void testInsertWithCollision_1();
void testDeleteWithCollision_1();
void testResizeWithCollision_1();
void testMixedWithCollision_1();
void n3SumTest();
void n3SumTestTimeRun();

int main() {
  testInsertWithCollision_1();
  testDeleteWithCollision_1();
  testResizeWithCollision_1();
  testMixedWithCollision_1();
  n3SumTest();
  n3SumTestTimeRun();
}

void testInsertWithCollision_1() {
  cout << "Test Insert With Collision" << endl;

  HashTable ht1(23);

  int i = 1;
  // insert 1 - 9
  while (i < 10) {
    ht1.insert(i++);
  }

  i = 1;
  // insert again (should not do anything since already exist)
  while (i < 10) {
    ht1.insert(i++);
  }
  ht1.insert(44);  // sum of digits is 8
  ht1.insert(555); // sum of digits is 15
  ht1.insert(96);  // sum of digits is 15
  ht1.insert(11);  // sum of digits is 2

  cout << "Your output" << endl;
  ht1.printHashTable();

  cout << "Expected output" << endl;
  cout << "0 1 2 3 4 5 6 7 8 9 0 11 44 0 0 555 96 0 0 0 0 0 0 " << endl;
  cout << endl;
}

void testDeleteWithCollision_1() {
  cout << "Test Delete With Collision" << endl;

  HashTable ht1(23);

  int i = 1;

  // insert 1 - 9
  while (i < 10) {
    ht1.insert(i++);
  }
  ht1.insert(44);  // sum of digits is 8
  ht1.insert(555); // sum of digits is 15
  ht1.insert(96);  // sum of digits is 15
  ht1.insert(11);  // sum of digits is 2

  ht1.remove(1);
  ht1.remove(44);
  ht1.remove(555);
  ht1.remove(96);
  ht1.remove(9876); // item which doesn't exist

  cout << "Your output" << endl;
  ht1.printHashTable();
  cout << "Expected output should be:" << endl;
  cout << "0 -1 2 3 4 5 6 7 8 9 0 11 -1 0 0 -1 -1 0 0 0 0 0 0 " << endl;

  cout << endl;
}

void testResizeWithCollision_1() {
  cout << "Resized Test" << endl;
  HashTable ht1(23);

  ht1.insert(17);
  ht1.insert(26);
  ht1.insert(35);
  ht1.insert(44);
  ht1.insert(53);
  ht1.insert(62);
  ht1.insert(71);
  ht1.insert(80);
  ht1.insert(134);
  ht1.printHashTable();

  ht1.resize(31);
  cout << "After resize" << endl;
  ht1.printHashTable();
  cout << "Expected output" << endl;
  cout << "0 0 80 0 0 0 0 0 53 134 71 0 17 35 0 0 0 26 0 0 0 0 0 0 62 0 44 0 0 "
          "0 0 "
       << endl;

  cout << endl;
}

void testMixedWithCollision_1() {
  cout << "Mixed Test With Collision" << endl;

  HashTable ht1(23);

  int i = 1;

  // insert 1 - 9
  while (i < 10) {
    ht1.insert(i++);
  }

  ht1.printHashTable();
  ht1.insert(44);  // sum of digits is 8
  ht1.insert(555); // sum of digits is 15
  ht1.insert(96);  // sum of digits is 15
  ht1.insert(11);  // sum of digits is 2
  ht1.printHashTable();

  ht1.remove(1);
  ht1.remove(44);
  ht1.remove(5);
  ht1.remove(3);
  ht1.remove(9876); // doesn't exist in HT
  ht1.printHashTable();
  ht1.insert(71);
  ht1.insert(101);

  ht1.printHashTable();

  ht1.resize(17);
  ht1.printHashTable();
  ht1.resize(13);
  ht1.printHashTable();
  ht1.resize(11);
  ht1.printHashTable();
  ht1.resize(29);
  ht1.printHashTable();

  cout << "Expected final output should be:" << endl;
  cout << "0 0 11 2 4 0 101 6 71 8 9 7 0 0 0 96 555 0 0 0 0 0 0 0 0 0 0 0 0 "
       << endl;
}

void n3SumTest() {
  cout << "n3Sum Test" << endl;
  int test1[] = {14, 52, 23, 11, 12, 72, 21, 22, 13, 53, 54};
  cout << n3Sum(test1, 10, 90) << endl; // (14, 23, 53)
  cout << n3Sum(test1, 10, 89)
       << endl; // (14, 52, 23), (14, 22, 53), (13, 53, 23)
  cout << n3Sum(test1, 10, 88) << endl; // (14, 52, 22), (14, 21, 53), (52, 23,
                                        // 13), (52, 22, 14), (12, 53, 23)
}

void n3SumTestTimeRun() {
  clock_t begin1, end1;

  double time_spent1;

  const int N = 4000;
  int test[N];
  for (int i = 0; i < N; i++) {
    test[i] = i * 2 + 1;
  }

  for (int a = 500; a <= 3500; a += 1000) {
    begin1 = clock();
    cout << "Ans for " << a << " = " << n3Sum(test, a, 300 * a / 400) << endl;
    end1 = clock();

    time_spent1 = (double)(end1 - begin1) / CLOCKS_PER_SEC;

    cout << "Time: " << time_spent1 << "s" << endl;
  }

  /* Expected answers and timings (roughly)
   * Ans for 500 = 2883
   * Time: 0.001356s
   * Ans for 1500 = 26227
   * Time: 0.013978s
   * Ans for 2500 = 73008
   * Time: 0.043073s
   * Ans for 3500 = 143227
   * Time: 0.098489s
   */
}
