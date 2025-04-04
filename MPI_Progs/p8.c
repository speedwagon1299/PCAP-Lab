// 4x4 search

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define MCW MPI_COMM_WORLD

void callErr(int err) {
    if(err != 0) {
        char estr[50]; int len = 50;
        MPI_Error_string(err, estr, &len);
        printf("\nError: %s", estr);
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(MCW, MPI_ERRORS_RETURN);
    int rank, size, err;
    MPI_Comm_rank(MCW, &rank);
    MPI_Comm_size(MCW, &size);
    MPI_Status stat;
    int mat[size][size], temp[size], search, occ, totocc;
    if(rank == 0) {
        printf("\nEnter %d x %d elements:\n", size, size);
        for(int i = 0; i < size; i++) {
            for(int j = 0; j < size; j++) {
                scanf("%d", &mat[i][j]);
            }
        }
        printf("\nThe Matrix:\n");
        for(int i = 0; i < size; i++) {
            for(int j = 0; j < size; j++) {
                printf("%d ", mat[i][j]);
            }
            printf("\n");
        }
        printf("\n\nEnter the element to be searched:\n");
        scanf("%d",&search);
    }
    err = MPI_Bcast(&search, 1, MPI_INT, 0, MCW);
    callErr(err);
    err = MPI_Scatter(mat, size, MPI_INT, temp, size, MPI_INT, 0, MCW);
    callErr(err);
    occ = 0; totocc = 0;
    for(int i = 0; i < size; i++) {
        if(temp[i] == search) {
            occ++;
        }
    }
    printf("\nRank %d: %d Occurences", rank, occ);
    // err = MPI_Reduce(&occ, &totocc, 1, MPI_INT, MPI_SUM, 0, MCW);
    err = MPI_Scan(&occ, &totocc, 1, MPI_INT, MPI_SUM, MCW);
    callErr(err);
    printf("\n\nTotal Occurences: %d", totocc);
    MPI_Finalize();
}