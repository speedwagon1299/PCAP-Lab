// reverse sentence prog

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define MCW MPI_COMM_WORLD

void reverse(char* buf) {
    int len = strlen(buf);
    for(int i = 0; i < len/2; i++) {
        char temp = buf[i];
        buf[i] = buf[len-i-1];
        buf[len-i-1] = temp;
    }
}

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MCW, &rank);
    MPI_Comm_size(MCW, &size);
    MPI_Status stat;
    char sent[size][50];
    char buf[50];
    if(rank == 0) {
        for(int i = 1; i < size; i++) {
            scanf("%s ", &sent[i]);
            MPI_Ssend(sent[i], 50, MPI_CHAR, i, 0, MCW);
        }
    }
    else {
        MPI_Recv(buf, 50, MPI_CHAR, 0, 0, MCW, &stat);
        reverse(buf);
    }
    MPI_Gather(buf, 50, MPI_CHAR, sent, 50, MPI_CHAR, 0, MCW);
    if(rank == 0) {
        for(int i = 1; i < size; i++) {
            printf("%s ", sent[i]);
        }
    }
    MPI_Finalize();
}