#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#define SIZE 200
int main (int argc, char ** argv)  {
	int input[SIZE];
	int prefix[SIZE] = {0};

	// Only rank 0 initializes the array
	for (int i = 0; i < SIZE; i++) 
		input[i] = i;

	for (int i = 0; i < SIZE; i++) {
		int prev_num = i != 0 ? prefix[i - 1] : 0;
		prefix[i] = prev_num + input[i];
	}

	printf("Prefix sums: ");
	for (int i = 0; i < SIZE; i++) {
		printf("%d, ", prefix[i]);
	}
	printf("\n");

	return 0;
}

