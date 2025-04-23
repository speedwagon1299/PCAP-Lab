#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void toggle(char* word, int i) {
	if(i < strlen(word)) {
		if(word[i] >= 'A' && word[i] <= 'Z') {
			word[i] += 32;
		}
		else {
			word[i] -= 32;
		}
	}
	printf("%s\n", word);
}

int main(int argc, char* argv[]) {
	int rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	char* word = argv[1];
	toggle(word, rank);
	MPI_Finalize();
	exit(0);
}