// read two strings s1 and s2 of same length in root
// alternatively merge

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#define MCW MPI_COMM_WORLD
#define ELEN 50

void callErr(int err) {
    if(err != 0) {
        char estr[ELEN]; int len = ELEN;
        MPI_Error_string(err, estr, &len);
        printf("\nError: %s", estr);
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size, err, len;
    MPI_Comm_set_errhandler(MCW, MPI_ERRORS_RETURN);  // MPI_Errhandler_set
    MPI_Comm_rank(MCW, &rank);
    MPI_Comm_size(MCW, &size);
    MPI_Status stat;
    char s1[50], s2[50], sub1[20], sub2[10], final[100];
    if(rank == 0) {
        printf("Enter the two strings s1 and s2:\n");
        scanf("%s%s", s1, s2);
        len = strlen(s1);
    }
    err = MPI_Bcast(&len, 1, MPI_INT, 0, MCW);    
    callErr(err);
    err = MPI_Scatter(s1, len/size, MPI_CHAR, sub1, len/size, MPI_CHAR, 0, MCW);
    err = MPI_Scatter(s2, len/size, MPI_CHAR, sub2, len/size, MPI_CHAR, 0, MCW);
    strcat(sub1, sub2);
    err = MPI_Gather(sub1, 2*len/size, MPI_CHAR, final, 2*len/size, MPI_CHAR, 0, MCW);
    if(rank == 0) {
        final[2*len] = '\0';
        printf("\nFinal Sentence: %s", final);
    }
    MPI_Finalize();
}