#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define X 5

int main(int argc, char* argv[]) {
	int rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int val = pow(X, rank);
	printf("My rank is %d and power is %d\n", rank, val);
	MPI_Finalize();
	exit(0);
}