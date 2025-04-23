#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
	int rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if(rank%2 == 0) {
		printf("Hello\n");
	}
	else {
		printf("World\n");
	}
	MPI_Finalize();
	exit(0);
}