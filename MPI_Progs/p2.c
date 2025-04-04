// circular string toggle

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#define MCW MPI_COMM_WORLD

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size, err;
    MPI_Comm_rank(MCW, &rank);
    MPI_Comm_size(MCW, &size);
    MPI_Status stat;
    char word[50];
    if(rank == 0) {
        printf("Enter the String:\n");
        fgets(word, 50, stdin);
        word[size] = '\0';
        MPI_Send(word, size + 1, MPI_CHAR, 1, 0, MCW);
        MPI_Recv(word, size + 1, MPI_CHAR, size - 1, 0, MCW, &stat);
        toggle(word);
        printf("Rank %d: %s\n", rank, word);
    }
    else {
        MPI_Recv(word, size + 1, MPI_CHAR, rank - 1, 0, MCW, &stat);
        toggle(word);
        printf("Rank %d: %s\n", rank, word);
        MPI_Send(word, size + 1, MPI_CHAR, (rank + 1) % size, 0, MCW);
    }
    MPI_Finalize();
}