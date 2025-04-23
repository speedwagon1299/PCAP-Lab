#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <string.h>
#define WORD_LEN 40
#define MCW MPI_COMM_WORLD

void hErr(int err) {
    if(err != MPI_SUCCESS) {
        int len = 50, ecode; char estr[len];
        MPI_Error_class(err, &ecode);
        MPI_Error_string(err, estr, &len);
        printf("\nError Code:\t%d\nError: %s\n", ecode, estr);
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(MCW, MPI_ERRORS_RETURN);
    int rank, size, err;
    MPI_Comm_rank(MCW, &rank);
    MPI_Comm_size(MCW, &size);
    MPI_Status stat;
    char sent[size][WORD_LEN], word[WORD_LEN];
    if(rank == 0) {
        printf("\nEnter %d words:\t", size);
        fflush(stdout);
        for(int i = 0; i < size; i++) {
            scanf("%s", sent[i]);
        }
    }
    err = MPI_Scatter(sent, WORD_LEN, MPI_CHAR, word, WORD_LEN, MPI_CHAR, 0, MCW);
    hErr(err);
    int occ = 0, res = 0;
    for(int i = 0; word[i] != '\0'; i++) {
        if(strchr("aeiouAEIOU",word[i]) != NULL) {
            occ++;
        }
    }
    err = MPI_Reduce(&occ, &res, 1, MPI_INT, MPI_SUM, 0, MCW);
    hErr(err);
    if(rank == 0) {
        printf("\nNumber of Vowels: %d", res);
        fflush(stdout);
    }
    MPI_Finalize();
}