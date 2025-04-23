#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char* argv[]) {
	int rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int x = atoi(argv[1]);
	int y = atoi(argv[2]);
	if(rank%5 == 0) {
		printf("x + y = %d\n", x+y);
	}
	else if(rank%5 == 1) {
		printf("x - y = %d\n", x-y);
	}
	else if(rank%5 == 2) {
		printf("x * y = %d\n", x*y);
	}
	else if(rank%5 == 3) {
		printf("x % y = %d\n", x%y);
	}
	else {
		printf("x / y = %d\n", x/y);
	}
	MPI_Finalize();
	exit(0);
}
