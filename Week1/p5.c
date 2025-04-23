#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

int fibo(int num) {
	if(num <= 1) {
		return num;
	}
	return fibo(num-1) + fibo(num-2);
}

int fact(int num) {
	if(num <= 1) {
		return 1;
	}
	return num * fact(num-1);
}

int main(int argc, char* argv[]) {
	int rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if(rank%2 == 0) {
		printf("Rank: %d\tFactorial: %d\n", rank, fact(rank));
	}
	else {
		printf("Rank: %d\tFibonacci: %d\n", rank, fibo(rank));
	}
	MPI_Finalize();
	exit(0);
}